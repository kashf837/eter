// ETER — NURBS real-time rendering
// Architecture: NURBS .surf → knot insertion → Bezier patches → GPU eval
//              → Vulkan triangle mesh → dynamic rendering → ImGui display
#define VMA_IMPLEMENTATION

#pragma region Includes
#include <vector>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <backends/imgui_impl_vulkan.h>

#include "nurbs.h"  // nurbs_wireframe()
#include <nvapp/application.hpp>
#include <nvapp/elem_profiler.hpp>
#include <nvapp/elem_logger.hpp>
#include <nvapp/elem_camera.hpp>
#include <nvgui/file_dialog.hpp>
#include <nvutils/logger.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/context.hpp>
#include <nvvk/default_structs.hpp>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/sampler_pool.hpp>
#include <nvvk/staging.hpp>
#include <nvvk/profiler_vk.hpp>
#include <nvutils/parameter_parser.hpp>
#pragma endregion

// ============================================================
// Utilities
// ============================================================

// Read binary SPIR-V shader file into memory
static std::vector<char> readFile(const std::string& fn) {
    std::ifstream f(fn, std::ios::ate | std::ios::binary);
    if (!f) return {};
    size_t sz = f.tellg(); std::vector<char> buf(sz);
    f.seekg(0); f.read(buf.data(), sz);
    return buf;
}

// MVP matrices passed to vertex shader via uniform buffer
// (matches layout(set=0, binding=0) in mesh.vert)
struct MVP { glm::mat4 model, view, proj; float intensity, lightAz, lightEl; float _pad; };

// ============================================================
// Mesh — host-side triangle mesh data
// ============================================================
struct Mesh {
    std::vector<float>    v;  // interleaved xyz + nxnynz (6 floats/vertex)
    std::vector<uint32_t> i;  // triangle indices
    float bbMin[3], bbMax[3];
    bool   empty() const { return i.empty(); }
    int    verts() const { return (int)(v.size()/6); }
    float  scale() const { float dx=bbMax[0]-bbMin[0],dy=bbMax[1]-bbMin[1],dz=bbMax[2]-bbMin[2];
                           float s=dx>dy?dx:dy; s=s>dz?s:dz; return s<1.f?1.f:s; }
    float  cx() const { return (bbMin[0]+bbMax[0])*0.5f; }
    float  cy() const { return (bbMin[1]+bbMax[1])*0.5f; }
    float  cz() const { return (bbMin[2]+bbMax[2])*0.5f; }
};

// ============================================================
// MeshBuilder — builds Mesh from .surf file(s)
// ============================================================
class MeshBuilder {
public:

    static Mesh adaptive(const char* path, const CameraParams* cam) {
        Mesh m; resetBBox(m);
        int nu, nv, total;
        PatchInfo* info;
        float* verts = nurbs_adaptive_eval(path, cam, &nu, &nv, &info, &total);
        if (!verts) { fprintf(stderr, "  [SKIP] %s\n", path); return m; }
        for (int pi = 0; pi < nu; pi++)
            for (int pj = 0; pj < nv; pj++) {
                PatchInfo* p = &info[pi*nv+pj];
                float* patch = verts + p->offset * 6;
                uint32_t base = (uint32_t)m.v.size() / 6;
                for (int y = 0; y < p->tv; y++)
                    for (int x = 0; x < p->tu; x++) {
                        int k = y*p->tu + x;
                        float px=patch[k*6+0], py=patch[k*6+1], pz=patch[k*6+2];
                        float nx=patch[k*6+3], ny=patch[k*6+4], nz=patch[k*6+5];
                        float nl=nx*nx+ny*ny+nz*nz;
                        if (nl < 1e-6f) {
                            int xR=(x+1<p->tu)?x+1:x, xL=xR>0?xR-1:0;
                            int yD=(y+1<p->tv)?y+1:y, yU=yD>0?yD-1:0;
                            float du[3]={patch[(y*p->tu+xR)*6+0]-patch[(y*p->tu+xL)*6+0],patch[(y*p->tu+xR)*6+1]-patch[(y*p->tu+xL)*6+1],patch[(y*p->tu+xR)*6+2]-patch[(y*p->tu+xL)*6+2]};
                            float dv[3]={patch[(yD*p->tu+x)*6+0]-patch[(yU*p->tu+x)*6+0],patch[(yD*p->tu+x)*6+1]-patch[(yU*p->tu+x)*6+1],patch[(yD*p->tu+x)*6+2]-patch[(yU*p->tu+x)*6+2]};
                            nx=du[1]*dv[2]-du[2]*dv[1];ny=du[2]*dv[0]-du[0]*dv[2];nz=du[0]*dv[1]-du[1]*dv[0];
                            nl=nx*nx+ny*ny+nz*nz;
                        }
                        float inv=1.f/(sqrtf(nl)+1e-10f);
                        m.v.insert(m.v.end(),{px,py,pz,nx*inv,ny*inv,nz*inv});
                        if(px<m.bbMin[0])m.bbMin[0]=px;if(py<m.bbMin[1])m.bbMin[1]=py;if(pz<m.bbMin[2])m.bbMin[2]=pz;
                        if(px>m.bbMax[0])m.bbMax[0]=px;if(py>m.bbMax[1])m.bbMax[1]=py;if(pz>m.bbMax[2])m.bbMax[2]=pz;
                    }
                for (int y = 0; y < p->tv-1; y++)
                    for (int x = 0; x < p->tu-1; x++) {
                        uint32_t a=base+y*p->tu+x, b=a+1, c=a+p->tu, d=c+1;
                        m.i.insert(m.i.end(),{a,b,c,b,d,c});
                    }
            }
        free(verts); free(info);
        LOGI("Adaptive %s: %d verts, %d tris\n", path, m.verts(), (int)(m.i.size()/3));
        return m;
    }

    static Mesh single(const char* path, int res = 32) {
        Mesh m; resetBBox(m);
        int nu, nv;
        float* verts = nurbs_wireframe(path, res, &nu, &nv);
        if (!verts) { fprintf(stderr, "  [SKIP] %s\n", path); return m; }
        addPatchGrid(m, verts, nu, nv, res);
        free(verts);
        LOGI("Loaded %s: %d verts, %d tris\n", path, m.verts(), (int)(m.i.size()/3));
        return m;
    }

    static Mesh directory(const char* dirPath, int res = 32) {
        namespace fs = std::filesystem;
        Mesh m; resetBBox(m);
        std::vector<std::string> files;
        for (auto& e : fs::directory_iterator(dirPath))
            if (e.path().extension() == ".surf") files.push_back(e.path().string());
        std::sort(files.begin(), files.end());
        LOGI("Loading %zu surf files (res=%d)...\n", files.size(), res);
        int skipped = 0;
        for (size_t i = 0; i < files.size(); i++) {
            if ((i+1) % 500 == 0) LOGI("  [%zu/%zu] ...\n", i+1, files.size());
            int nu, nv;
            float* verts = nurbs_wireframe(files[i].c_str(), res, &nu, &nv);
            if (!verts) { skipped++; continue; }
            addPatchGrid(m, verts, nu, nv, res);
            free(verts);
        }
        LOGI("Done: %d verts, %d tris, %d skipped\n", m.verts(), (int)(m.i.size()/3), skipped);
        return m;
    }
private:

    static void resetBBox(Mesh& m) {
        m.bbMin[0]=m.bbMin[1]=m.bbMin[2]= 1e30f;
        m.bbMax[0]=m.bbMax[1]=m.bbMax[2]=-1e30f;
    }

    static void addPatchGrid(Mesh& m, const float* verts, int nu, int nv, int res) {
        for (int pi = 0; pi < nu; pi++)
            for (int pj = 0; pj < nv; pj++) {
                uint32_t base = (uint32_t)m.v.size() / 6;
                const float* p = verts + (pi*nv+pj)*res*res*6;
                for (int y = 0; y < res; y++)
                    for (int x = 0; x < res; x++) {
                        int idx = y*res+x;
                        float px=p[idx*6+0], py=p[idx*6+1], pz=p[idx*6+2];
                        float gnx=p[idx*6+3], gny=p[idx*6+4], gnz=p[idx*6+5];
                        float gnLen = gnx*gnx+gny*gny+gnz*gnz;
                        if(gnLen < 1e-6f) {
                            // CPU fallback (shared-memory kernel)
                            int xR=(x+1<res)?x+1:x,xL=xR>0?xR-1:0, yD=(y+1<res)?y+1:y,yU=yD>0?yD-1:0;
                            float du[3]={p[(y*res+xR)*6+0]-p[(y*res+xL)*6+0],p[(y*res+xR)*6+1]-p[(y*res+xL)*6+1],p[(y*res+xR)*6+2]-p[(y*res+xL)*6+2]};
                            float dv[3]={p[(yD*res+x)*6+0]-p[(yU*res+x)*6+0],p[(yD*res+x)*6+1]-p[(yU*res+x)*6+1],p[(yD*res+x)*6+2]-p[(yU*res+x)*6+2]};
                            gnx=du[1]*dv[2]-du[2]*dv[1]; gny=du[2]*dv[0]-du[0]*dv[2]; gnz=du[0]*dv[1]-du[1]*dv[0];
                            gnLen = gnx*gnx+gny*gny+gnz*gnz;
                        }
                        float inv=1.f/(sqrtf(gnLen)+1e-10f);
                        m.v.insert(m.v.end(),{px,py,pz, gnx*inv,gny*inv,gnz*inv});
                        if(px<m.bbMin[0])m.bbMin[0]=px; if(py<m.bbMin[1])m.bbMin[1]=py; if(pz<m.bbMin[2])m.bbMin[2]=pz;
                        if(px>m.bbMax[0])m.bbMax[0]=px; if(py>m.bbMax[1])m.bbMax[1]=py; if(pz>m.bbMax[2])m.bbMax[2]=pz;
                    }
                for (int y = 0; y < res-1; y++)
                    for (int x = 0; x < res-1; x++) {
                        uint32_t a=base+y*res+x, b=a+1, c=a+res, d=c+1;
                        m.i.insert(m.i.end(),{a,b,c, b,d,c});
                    }
            }
    }
};

// ============================================================
// NURBSRenderer — implements IAppElement, renders NURBS surface as 3D mesh
// ============================================================
// The nvpro_core2 framework uses a "plugin" architecture:
//   Application owns the window + main loop.
//   Each IAppElement gets callbacks at specific times:
//     onAttach()  — once at startup: create GPU resources
//     onRender()  — every frame: record GPU draw commands
//     onDetach()  — once at shutdown: destroy GPU resources
//     onUIRender()— every frame: draw ImGui panels

class NURBSRenderer : public nvapp::IAppElement
{
public:
  struct Info { nvutils::ProfilerManager* profilerManager{}; nvutils::ParameterRegistry* parameterRegistry{}; };
  NURBSRenderer(const Info& info) : m_info(info) { }

  // ---- Expose camera so ElementCamera can control it ----
  auto getCamera() { return m_camera; }

  // Reload mesh from a single file or directory (STEP temp)
  void doReload(const std::string& path, bool isDir = false) {
    LOGI("doReload: %s (dir=%d)\n", path.c_str(), isDir);
    NVVK_CHECK(vkDeviceWaitIdle(m_app->getDevice()));
    if (m_indexCount > 0) {
        m_alloc.destroyBuffer(m_indexBuffer);
        m_alloc.destroyBuffer(m_vertexBuffer);
        m_indexCount = 0;
    }
    Mesh m;
    if (isDir) { m = MeshBuilder::directory(path.c_str(), 32); }
    else {
        CameraParams cam;
        glm::dvec3 eye = m_camera->getEye();
        if (eye.x > 5.f || eye.x < -5.f) eye = {0.f, 0.f, -2.f};
        cam.camX=(float)eye.x; cam.camY=(float)eye.y; cam.camZ=(float)eye.z;
        cam.fovY=60.f; cam.scrW=m_imgW; cam.scrH=m_imgH;
        m = MeshBuilder::adaptive(path.c_str(), &cam);
    }

    if (!m.empty()) {
        uploadMesh(m);
        m_modelScale = m.scale();
        m_center[0]=m.cx(); m_center[1]=m.cy(); m_center[2]=m.cz();
        float dist = m_modelScale * 1.5f;
        m_camera->setLookat({m_center[0],m_center[1],m_center[2]+dist},
                            {m_center[0],m_center[1],m_center[2]},{0,-1,0});
        m_camera->setWindowSize({(uint32_t)m_imgW,(uint32_t)m_imgH});
        LOGI("Camera: center=(%.1f,%.1f,%.1f) scale=%.1f dist=%.1f\n",
             m_center[0],m_center[1],m_center[2],m_modelScale,dist);
      glm::dvec3 e = m_camera->getEye();
      m_lastEye[0]=(float)e.x; m_lastEye[1]=(float)e.y; m_lastEye[2]=(float)e.z;
      m_lastPath = path;
    }
    if (isDir) std::filesystem::remove_all(path);
  }

  // ============================================================
  // onAttach — one-time GPU resource creation
  // ============================================================
  void onAttach(nvapp::Application* app) override
  {
    m_app = app;
    VkDevice dev = app->getDevice();
    initAllocator(app);

    // 1. Create images that we render into and display
    createViewportImage(dev);   // ImGui shows this texture
    createRenderTarget(dev);    // we draw 3D mesh here, then blit→viewport
    createDepthBuffer(dev);     // for correct occlusion (Z-buffer)

    // 2. Create Vulkan graphics pipeline (shaders + state)
    createPipeline(dev);

    // 3. Create descriptor set — connects UBO (MVP matrix) to shader
    createDescriptorSet(dev);

    // 4. Upload MVP matrix (updated per-frame by camera)
    createUniformBuffer(dev);

    // 5. Register viewport image with ImGui (so onUIRender can display it)
    initViewportForImGui(dev);

    // 6b. Auto-fit camera to loaded model
    if (m_indexCount > 0) {
        float dist = m_modelScale * 1.5f;
        m_camera->setLookat({m_center[0],m_center[1],m_center[2]+dist},
                            {m_center[0],m_center[1],m_center[2]},{0,-1,0});
        m_camera->setWindowSize({(uint32_t)m_imgW,(uint32_t)m_imgH});
        LOGI("Camera: center=(%.1f,%.1f,%.1f) scale=%.1f dist=%.1f\n",
             m_center[0],m_center[1],m_center[2],m_modelScale,dist);
    }

    // 7. Profiler
    m_profilerTimeline = m_info.profilerManager->createTimeline({"graphics"});
    m_profilerGpuTimer.init(m_profilerTimeline, dev,
        app->getPhysicalDevice(), app->getQueue(0).familyIndex, true);

  }

  // ============================================================
  // onDetach — destroy all GPU resources (reverse order of creation)
  // ============================================================
  void onDetach() override
  {
    NVVK_CHECK(vkDeviceWaitIdle(m_app->getDevice()));  // wait for GPU to finish
    VkDevice dev = m_app->getDevice();
    ImGui_ImplVulkan_RemoveTexture(m_imguiImage);

    // Pipeline
    vkDestroyPipeline(dev, m_pipeline, nullptr);
    vkDestroyPipelineLayout(dev, m_pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(dev, m_dsLayout, nullptr);
    vkDestroyDescriptorPool(dev, m_dsPool, nullptr);

    // Buffers & images
    vmaDestroyBuffer((VmaAllocator)m_alloc, m_ubo.buffer, m_ubo.allocation);
    m_alloc.destroyBuffer(m_indexBuffer);
    m_alloc.destroyBuffer(m_vertexBuffer);
    m_alloc.destroyImage(m_depthImage);
    m_alloc.destroyImage(m_renderTarget);
    m_alloc.destroyImage(m_viewportImage);

    // Framework
    m_stagingUploader.deinit();
    m_samplerPool.deinit();
    m_alloc.deinit();
    m_profilerGpuTimer.deinit();
    m_info.profilerManager->destroyTimeline(m_profilerTimeline);
  }

  // ============================================================
  // onRender — called every frame: draw 3D mesh → blit→ viewport
  // ============================================================
  void onRender(VkCommandBuffer cmd) override
  {
    // Process pending mesh reload (deferred from menu callback)
    if (!m_pendingLoad.empty()) {
        doReload(m_pendingLoad, m_pendingIsDir);
        m_pendingLoad.clear();
        return;  // skip rendering this frame (buffers were recreated)
    }
    // Upload latest camera matrices to GPU
    updateCameraUBO();

    // Step A: transition render target + depth to writable layouts
    VkImageLayout rtOld = m_firstFrame ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    m_firstFrame = false;
    if (rtOld == VK_IMAGE_LAYOUT_UNDEFINED) {
        nvvk::cmdImageMemoryBarrier(cmd, {m_renderTarget.image,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});
        VkImageMemoryBarrier2 db = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        db.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        db.dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT;
        db.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        db.newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        db.image = m_depthImage.image;
        db.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
        VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &db;
        vkCmdPipelineBarrier2(cmd, &di);
    }

    // Step B: begin dynamic rendering (Vulkan 1.3 core feature — no render pass object needed)
    VkRenderingAttachmentInfo ca = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    ca.imageView   = m_renderTarget.descriptor.imageView;
    ca.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    ca.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
    ca.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
    ca.clearValue.color = {0.1f, 0.1f, 0.15f, 1.f};

    VkRenderingAttachmentInfo da = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    da.imageView   = m_depthImage.descriptor.imageView;
    da.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    da.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
    da.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
    da.clearValue.depthStencil = {1.f, 0};

    VkRenderingInfo ri = {VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = {{0, 0}, {(uint32_t)m_imgW, (uint32_t)m_imgH}};
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1; ri.pColorAttachments = &ca;
    ri.pDepthAttachment = &da;
    vkCmdBeginRendering(cmd, &ri);

    // Step C: draw the mesh (skip if nothing loaded)
    if (m_indexCount > 0) {
    VkViewport vp = {0, (float)m_imgH, (float)m_imgW, -(float)m_imgH, 0, 1}; // Vulkan Y-flip
    VkRect2D   sc = {{0, 0}, {(uint32_t)m_imgW, (uint32_t)m_imgH}};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    vkCmdSetScissor(cmd, 0, 1, &sc);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, &m_ds, 0, nullptr);
    VkDeviceSize off = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &m_vertexBuffer.buffer, &off);
    vkCmdBindIndexBuffer(cmd, m_indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cmd, m_indexCount, 1, 0, 0, 0);
    } // m_indexCount > 0
    vkCmdEndRendering(cmd);

    // Step D: blit (copy) render target → viewport image for ImGui display
    nvvk::cmdImageMemoryBarrier(cmd, {m_renderTarget.image,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL});
    nvvk::cmdImageMemoryBarrier(cmd, {m_viewportImage.image,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL});
    VkImageBlit blit = {};
    blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.srcOffsets[1]  = {m_imgW, m_imgH, 1};
    blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.dstOffsets[1]  = {m_imgW, m_imgH, 1};
    vkCmdBlitImage(cmd, m_renderTarget.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   m_viewportImage.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_NEAREST);

    // Step E: transition images back for next frame
    nvvk::cmdImageMemoryBarrier(cmd, {m_renderTarget.image,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});
    nvvk::cmdImageMemoryBarrier(cmd, {m_viewportImage.image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL});
  }

  // ============================================================
  // onUIRender — ImGui panels
  // ============================================================
  void onUIRender() override
  {
    ImGui::Begin("Settings");
    ImGui::SliderFloat("Intensity", &m_intensity, 0.0f, 2.0f);
    ImGui::SliderFloat("Light Azimuth", &m_lightAz, -3.14f, 3.14f);
    ImGui::SliderFloat("Light Elevation", &m_lightEl, 0.0f, 1.57f);
    if (ImGui::Checkbox("Conservative Raster", &m_useConservative)) {
        if (!m_lastPath.empty()) { m_pendingLoad = m_lastPath; m_pendingIsDir = false; }
    }
    ImGui::Text("Triangles: %d", m_indexCount / 3);
    ImGui::TextDisabled("%d FPS / %.3fms",
        static_cast<int>(ImGui::GetIO().Framerate), 1000.f / ImGui::GetIO().Framerate);
    ImGui::Text("Viewport: %d x %d", m_app->getViewportSize().width, m_app->getViewportSize().height);
    ImGui::End();

    // Viewport: display our rendered image (render target → viewport via blit)
    ImGui::Begin("Viewport");
    ImGui::Image((ImTextureID)m_imguiImage, ImGui::GetContentRegionAvail());
    ImGui::End();
  }

  void onPreRender() override {
    m_profilerTimeline->frameAdvance();
    // Check if camera moved enough to re-tessellate
    if (!m_lastPath.empty() && m_indexCount > 0) {
        glm::dvec3 e = m_camera->getEye();
        float dx=(float)e.x-m_lastEye[0], dy=(float)e.y-m_lastEye[1], dz=(float)e.z-m_lastEye[2];
        float dist = sqrtf(dx*dx+dy*dy+dz*dz);
        if (dist > m_modelScale * 0.2f) {  // moved >20% of model size
            CameraParams cam;
            cam.camX=(float)e.x; cam.camY=(float)e.y; cam.camZ=(float)e.z;
            cam.fovY=60.f; cam.scrW=m_imgW; cam.scrH=m_imgH;
            int res = compute_adaptive_res(m_lastPath.c_str(), &cam);
            Mesh m = MeshBuilder::single(m_lastPath.c_str(), res);
            if (!m.empty()) {
                if (m_indexCount > 0) { m_alloc.destroyBuffer(m_indexBuffer); m_alloc.destroyBuffer(m_vertexBuffer); m_indexCount = 0; }
                uploadMesh(m);
                m_modelScale = m.scale();
                m_center[0]=m.cx(); m_center[1]=m.cy(); m_center[2]=m.cz();
                m_lastEye[0]=(float)e.x; m_lastEye[1]=(float)e.y; m_lastEye[2]=(float)e.z;
                LOGI("[Adaptive] camera moved, res=%d\n", res);
            }
        }
    }
  }

  // ============================================================
  // onUIMenu — menu bar (File / View)
  // ============================================================
  void onUIMenu() override
  {
    bool vsync = m_app->isVsync();
    if (ImGui::BeginMenu("File")) {
        if (ImGui::MenuItem("Open...")) {
            auto path = nvgui::windowOpenFileDialog(m_app->getWindowHandle(), "Open SURF / STEP", "*.surf;*.stp;*.step");
            if (!path.empty()) {
                std::string sp = path.string();
                if (sp.ends_with(".stp") || sp.ends_with(".step")) {
                    auto stem = path.stem().string();
                    m_tempDir = (path.parent_path() / (stem + "_parts")).string();
                    std::filesystem::create_directory(m_tempDir);
                    std::string cmd = "python scripts/step2surf.py --all \"" + sp + "\" \"" + m_tempDir + "/out\"";
                    LOGI("Converting STEP...\n"); system(cmd.c_str());
                    m_pendingLoad = m_tempDir; m_pendingIsDir = true;
                } else { m_pendingLoad = sp; m_pendingIsDir = false; }
            }
        }
        if (ImGui::MenuItem("Exit", "Ctrl+Q")) m_app->close();
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("View")) {
        ImGui::MenuItem("V-Sync", "Ctrl+Shift+V", &vsync);
        ImGui::EndMenu();
    }
    if (ImGui::IsKeyPressed(ImGuiKey_Q) && ImGui::IsKeyDown(ImGuiKey_LeftCtrl)) m_app->close();
    if (ImGui::IsKeyPressed(ImGuiKey_V) && ImGui::IsKeyDown(ImGuiKey_LeftCtrl) && ImGui::IsKeyDown(ImGuiKey_LeftShift))
        vsync = !vsync;
    if (vsync != m_app->isVsync()) m_app->setVsync(vsync);
  }

private:
#pragma region Framework
  Info                          m_info;
  std::string                   m_pendingLoad;
  std::string                   m_tempDir;
  bool                          m_pendingIsDir = false;  // set by menu, processed next frame
  float                         m_intensity = 1.0f;
  float                         m_lightAz   = 0.6f;   // azimuth (radians)
  float                         m_lightEl   = 0.8f;   // elevation (radians)
  bool                          m_useConservative = false;
  nvapp::Application*           m_app{};
  nvvk::ResourceAllocator       m_alloc{};        // GPU memory manager (wraps VMA)
  nvvk::SamplerPool             m_samplerPool{};
  nvvk::StagingUploader         m_stagingUploader{}; // CPU→GPU data transfer helper
  nvutils::ProfilerTimeline*    m_profilerTimeline{};
  nvvk::ProfilerGpuTimer        m_profilerGpuTimer;
#pragma endregion

#pragma region Display images
  // Rendering pipeline:  mesh → m_renderTarget → blit → m_viewportImage → ImGui → screen
  int            m_imgW = 1024, m_imgH = 1024;
  nvvk::Image    m_viewportImage{};   // R32G32B32A32, ImGui display texture
  nvvk::Image    m_renderTarget{};    // R8G8B8A8, color attachment
  nvvk::Image    m_depthImage{};      // D32_SFLOAT, depth attachment (Z-buffer)
  VkDescriptorSet m_imguiImage{};
  bool           m_firstFrame = true;
#pragma endregion

#pragma region Camera
  std::shared_ptr<nvutils::CameraManipulator> m_camera = std::make_shared<nvutils::CameraManipulator>();
#pragma endregion

#pragma region Mesh geometry
  nvvk::Buffer m_vertexBuffer{};   // vertex positions (x,y,z per vertex)
  nvvk::Buffer m_indexBuffer{};    // triangle indices (3 x uint32_t per triangle)
  uint32_t     m_indexCount = 0;
  float m_modelScale = 1.0f;
  float m_center[3] = {0,0,0};
  float m_lastEye[3] = {0,0,0};
  std::string m_lastPath;
#pragma endregion

#pragma region Graphics pipeline
  // Pipeline = shaders + rasterization state + blend state + depth state
  VkPipeline            m_pipeline       = VK_NULL_HANDLE;
  VkPipelineLayout      m_pipelineLayout = VK_NULL_HANDLE;
  bool                  m_conservativeRaster = false;  // extension support
  VkDescriptorSetLayout m_dsLayout       = VK_NULL_HANDLE;
  VkDescriptorPool      m_dsPool         = VK_NULL_HANDLE;
  VkDescriptorSet       m_ds             = VK_NULL_HANDLE;
  nvvk::Buffer          m_ubo{};  // Uniform Buffer Object — holds MVP matrix for shader
#pragma endregion

  // ============================================================
  // Initialization helpers (called from onAttach)
  // ============================================================

  void initAllocator(nvapp::Application* app) {
    VmaAllocatorCreateInfo ai = {};
    ai.flags          = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    ai.physicalDevice = app->getPhysicalDevice();
    ai.device         = app->getDevice();
    ai.instance       = app->getInstance();
    NVVK_CHECK(m_alloc.init(ai));
    m_samplerPool.init(app->getDevice());
    m_stagingUploader.init(&m_alloc, true);
  }

  // Viewport image: what ImGui displays. We never write to it directly;
  // instead we render to m_renderTarget, then blit → m_viewportImage.
  void createViewportImage(VkDevice dev) {
    VkImageCreateInfo ii = DEFAULT_VkImageCreateInfo;
    ii.extent = {(uint32_t)m_imgW, (uint32_t)m_imgH, 1};
    ii.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    ii.usage  = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    // SAMPLED: ImGui can sample from it
    // TRANSFER_DST: vkCmdBlitImage can write to it
    VkImageViewCreateInfo vi = DEFAULT_VkImageViewCreateInfo;
    vi.components = {.a = VK_COMPONENT_SWIZZLE_ONE}; // force alpha=1 for display
    NVVK_CHECK(m_alloc.createImage(m_viewportImage, ii, vi));
    NVVK_CHECK(m_samplerPool.acquireSampler(m_viewportImage.descriptor.sampler));
  }

  // Render target: R8G8B8A8 (standard RGBA8), universally supports color attachment.
  void createRenderTarget(VkDevice dev) {
    VkImageCreateInfo ii = DEFAULT_VkImageCreateInfo;
    ii.extent = {(uint32_t)m_imgW, (uint32_t)m_imgH, 1};
    ii.format = VK_FORMAT_R8G8B8A8_UNORM;
    ii.usage  = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    // COLOR_ATTACHMENT: can render into it (vkCmdBeginRendering)
    // TRANSFER_SRC: can blit FROM it (vkCmdBlitImage)
    VkImageViewCreateInfo vi = DEFAULT_VkImageViewCreateInfo;
    NVVK_CHECK(m_alloc.createImage(m_renderTarget, ii, vi));
  }

  // Depth buffer: stores per-pixel depth (distance from camera).
  // GPU hardware uses this to decide which triangles are in front.
  // Without this, back triangles would draw over front triangles.
  void createDepthBuffer(VkDevice dev) {
    VkImageCreateInfo ii = DEFAULT_VkImageCreateInfo;
    ii.extent = {(uint32_t)m_imgW, (uint32_t)m_imgH, 1};
    ii.format = VK_FORMAT_D32_SFLOAT;  // 32-bit float depth
    ii.usage  = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    VkImageViewCreateInfo vi = DEFAULT_VkImageViewCreateInfo;
    vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    NVVK_CHECK(m_alloc.createImage(m_depthImage, ii, vi));
  }

  // Build Vulkan graphics pipeline from SPIR-V shaders.
  // The pipeline encapsulates: vertex shader, fragment shader, rasterization,
  // depth testing, blending, and dynamic rendering format info.
  void createPipeline(VkDevice dev) {
    // Check conservative rasterization support
    VkPhysicalDeviceConservativeRasterizationPropertiesEXT crProps = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONSERVATIVE_RASTERIZATION_PROPERTIES_EXT};
    VkPhysicalDeviceProperties2 props2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &crProps};
    vkGetPhysicalDeviceProperties2(m_app->getPhysicalDevice(), &props2);
    m_conservativeRaster = (crProps.primitiveOverestimationSize > 0.0f);
    LOGI("Conservative rasterization: %s\n", m_conservativeRaster ? "available" : "N/A");

    // Load compiled SPIR-V binaries
    auto vCode = readFile("shaders/spv/mesh.vert.spv");
    auto fCode = readFile("shaders/spv/mesh.frag.spv");
    if (vCode.empty() || fCode.empty()) { LOGE("Shader load failed\n"); return; }

    // Create temporary shader modules (destroyed after pipeline creation)
    VkShaderModuleCreateInfo sm = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    sm.codeSize = vCode.size(); sm.pCode = (const uint32_t*)vCode.data();
    VkShaderModule vs; vkCreateShaderModule(dev, &sm, nullptr, &vs);
    sm.codeSize = fCode.size(); sm.pCode = (const uint32_t*)fCode.data();
    VkShaderModule fs; vkCreateShaderModule(dev, &sm, nullptr, &fs);

    // Descriptor set layout: tells shader WHAT resources it can access.
    // Here: just one uniform buffer (MVP matrix) at set=0, binding=0.
    VkDescriptorSetLayoutBinding dslb = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT};
    VkDescriptorSetLayoutCreateInfo dslCI = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslCI.bindingCount = 1; dslCI.pBindings = &dslb;
    NVVK_CHECK(vkCreateDescriptorSetLayout(dev, &dslCI, nullptr, &m_dsLayout));

    VkPipelineLayoutCreateInfo plCI = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plCI.setLayoutCount = 1; plCI.pSetLayouts = &m_dsLayout;
    NVVK_CHECK(vkCreatePipelineLayout(dev, &plCI, nullptr, &m_pipelineLayout));

    // Vertex input: 6 floats per vertex (x,y,z + nx,ny,nz)
    VkVertexInputBindingDescription vib = {0, 6 * sizeof(float), VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription via[2] = {
        {0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0},           // location=0: position (offset 0)
        {1, 0, VK_FORMAT_R32G32B32_SFLOAT, 3*sizeof(float)}, // location=1: normal (offset 12)
    };
    VkPipelineVertexInputStateCreateInfo vi = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vi.vertexBindingDescriptionCount = 1; vi.pVertexBindingDescriptions = &vib;
    vi.vertexAttributeDescriptionCount = 2; vi.pVertexAttributeDescriptions = via;

    // Standard pipeline state: triangle list, 1 viewport, no blending
    VkPipelineInputAssemblyStateCreateInfo ia = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo vp = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, 0, 0, 1, nullptr, 1, nullptr};
    VkPipelineRasterizationStateCreateInfo rs = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.lineWidth = 1.f;
    VkPipelineRasterizationConservativeStateCreateInfoEXT cr = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_CONSERVATIVE_STATE_CREATE_INFO_EXT};
    cr.conservativeRasterizationMode = VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT;
    cr.extraPrimitiveOverestimationSize = 0.0f;
    if (m_useConservative) rs.pNext = &cr;
    VkPipelineMultisampleStateCreateInfo ms = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState cba = {}; cba.colorWriteMask = 0xF; // RGBA all channels
    VkPipelineColorBlendStateCreateInfo cb = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1; cb.pAttachments = &cba;

    // Depth test: enable Z-buffer (closer pixels occlude farther ones)
    VkPipelineDepthStencilStateCreateInfo dss = {VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    dss.depthTestEnable = VK_TRUE;
    dss.depthWriteEnable = VK_TRUE;
    dss.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

    // Dynamic state allows changing viewport/scissor without recreating pipeline
    VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo ds = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    ds.dynamicStateCount = 2; ds.pDynamicStates = dyn;

    VkPipelineShaderStageCreateInfo ss[] = {
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, 0, 0, VK_SHADER_STAGE_VERTEX_BIT, vs, "main"},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, 0, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fs, "main"},
    };

    // Dynamic rendering (Vulkan 1.3+): no VkRenderPass needed → simpler setup
    VkFormat cf = VK_FORMAT_R8G8B8A8_UNORM;
    VkFormat df = VK_FORMAT_D32_SFLOAT;
    VkPipelineRenderingCreateInfo pr = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    pr.colorAttachmentCount = 1; pr.pColorAttachmentFormats = &cf;
    pr.depthAttachmentFormat = df;

    VkGraphicsPipelineCreateInfo pi = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &pr};
    pi.stageCount = 2; pi.pStages = ss;
    pi.pVertexInputState = &vi; pi.pInputAssemblyState = &ia;
    pi.pViewportState = &vp; pi.pRasterizationState = &rs;
    pi.pMultisampleState = &ms; pi.pColorBlendState = &cb;
    pi.pDepthStencilState = &dss; pi.pDynamicState = &ds;
    pi.layout = m_pipelineLayout;

    NVVK_CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &pi, nullptr, &m_pipeline));
    vkDestroyShaderModule(dev, vs, nullptr);
    vkDestroyShaderModule(dev, fs, nullptr);
    LOGI("Pipeline created\n");
  }

  // Descriptor set: connects GPU resources (like the UBO) to shader bindings.
  // Think of it as "wiring" — the shader says "I need a UBO at binding 0",
  // and the descriptor set provides the actual buffer handle.
  void createDescriptorSet(VkDevice dev) {
    VkDescriptorPoolSize dps = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1};
    VkDescriptorPoolCreateInfo dpCI = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpCI.maxSets = 1; dpCI.poolSizeCount = 1; dpCI.pPoolSizes = &dps;
    NVVK_CHECK(vkCreateDescriptorPool(dev, &dpCI, nullptr, &m_dsPool));
    VkDescriptorSetAllocateInfo dsa = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsa.descriptorPool = m_dsPool; dsa.descriptorSetCount = 1; dsa.pSetLayouts = &m_dsLayout;
    NVVK_CHECK(vkAllocateDescriptorSets(dev, &dsa, &m_ds));
  }

  // Uniform buffer (UBO): GPU-visible memory holding the MVP matrices.
  // Created as host-visible + mapped, so we can memcpy() new values each frame
  // without any staging upload pipeline.
  void createUniformBuffer(VkDevice dev) {
    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = sizeof(MVP);
    bci.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    VmaAllocationCreateInfo aci = {};
    aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
              | VMA_ALLOCATION_CREATE_MAPPED_BIT;  // persistent CPU mapping
    aci.usage = VMA_MEMORY_USAGE_AUTO;
    VmaAllocationInfo ai;
    vmaCreateBuffer((VmaAllocator)m_alloc, &bci, &aci,
                    &m_ubo.buffer, &m_ubo.allocation, &ai);
    m_ubo.mapping    = (uint8_t*)ai.pMappedData;  // CPU can write here anytime
    m_ubo.bufferSize  = sizeof(MVP);

    // Write descriptor: tell shader "UBO is in this buffer at offset 0"
    VkDescriptorBufferInfo dbi = {m_ubo.buffer, 0, sizeof(MVP)};
    VkWriteDescriptorSet wds = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    wds.dstSet = m_ds; wds.dstBinding = 0; wds.descriptorCount = 1;
    wds.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; wds.pBufferInfo = &dbi;
    vkUpdateDescriptorSets(dev, 1, &wds, 0, nullptr);
  }

  void uploadMesh(const Mesh& m) {
    VkDevice dev = m_app->getDevice();
    VkDeviceSize vsz = m.v.size() * sizeof(float);
    m_alloc.createBuffer(m_vertexBuffer, vsz, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
    { void* p; NVVK_CHECK(m_stagingUploader.appendBufferMapping(m_vertexBuffer, 0, vsz, p)); memcpy(p, m.v.data(), vsz); }
    VkDeviceSize isz = m.i.size() * sizeof(uint32_t);
    m_alloc.createBuffer(m_indexBuffer, isz, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
    { void* p; NVVK_CHECK(m_stagingUploader.appendBufferMapping(m_indexBuffer, 0, isz, p)); memcpy(p, m.i.data(), isz); }
    VkCommandBuffer c = m_app->createTempCmdBuffer();
    m_stagingUploader.cmdUploadAppended(c);
    m_app->submitAndWaitTempCmdBuffer(c);
    m_stagingUploader.releaseStaging();
    m_indexCount = (uint32_t)m.i.size();
    LOGI("Uploaded: %d verts, %d tris\n", m.verts(), (int)(m.i.size()/3));
  }

  void initViewportForImGui(VkDevice dev) {
    VkCommandBuffer c = m_app->createTempCmdBuffer();
    nvvk::cmdImageMemoryBarrier(c, {m_viewportImage.image,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL});
    m_app->submitAndWaitTempCmdBuffer(c);
    m_imguiImage = ImGui_ImplVulkan_AddTexture(
        m_viewportImage.descriptor.sampler,
        m_viewportImage.descriptor.imageView,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  }

  // ============================================================
  // Per-frame helpers
  // ============================================================

  // Copy current camera view/proj matrices to GPU uniform buffer.
  // Called at the start of every onRender, before draw commands.
  void updateCameraUBO() {
    MVP mvp;
    mvp.model = glm::mat4(1.f);
    mvp.view  = m_camera->getViewMatrix();
    // Projection with viewport aspect ratio, not render-target aspect
    auto vsz  = m_app->getViewportSize();
    float asp = vsz.width / (float)vsz.height;
    float nearP = m_modelScale * 0.001f;  // auto-scale: near = 0.1% of model size
    float farP  = m_modelScale * 5.0f;     // far = 5× model size, plenty of room
    if (nearP < 0.001f) nearP = 0.001f;
    mvp.proj  = glm::perspective(glm::radians(60.f), asp, nearP, farP);
    mvp.proj[1][1] *= -1.f; // Vulkan Y flip
    mvp.intensity = m_intensity;
    mvp.lightAz   = m_lightAz;
    mvp.lightEl   = m_lightEl;
    memcpy(m_ubo.mapping, &mvp, sizeof(MVP));
  }
};

// ============================================================
// main() — application entry point
// ============================================================
int main(int argc, char** argv)
{
  nvutils::ProfilerManager   profilerManager;
  nvutils::ParameterRegistry parameterRegistry;
  nvutils::ParameterParser   parameterParser;

  // Create our rendering element
  NURBSRenderer::Info info = {&profilerManager, &parameterRegistry};
  auto nurbsRenderer = std::make_shared<NURBSRenderer>(info);

  // Logger element (shown by default in bottom panel)
  auto elementLogger = std::make_shared<nvapp::ElementLogger>(true);
  nvutils::Logger::getInstance().setLogCallback(
      [&](nvutils::Logger::LogLevel lvl, const std::string& text) {
          elementLogger->addLog(lvl, "%s", text.c_str());
      });

  // Vulkan context: instance + device + queues
  nvvk::ContextInitInfo vkSetup;
  vkSetup.instanceExtensions = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME};
  vkSetup.deviceExtensions   = {{VK_KHR_SWAPCHAIN_EXTENSION_NAME}, {VK_EXT_CONSERVATIVE_RASTERIZATION_EXTENSION_NAME, nullptr, true}};
  parameterRegistry.add({"validation"}, &vkSetup.enableValidationLayers);
  parameterRegistry.add({"verbose"},   &vkSetup.verbose);
  parameterRegistry.add({"forcedevice"}, &vkSetup.forceGPU);
  parameterParser.add(parameterRegistry);
  parameterParser.parse(argc, argv);

  nvvk::addSurfaceExtensions(vkSetup.instanceExtensions); // platform-specific surface ext
  nvvk::Context vkContext;
  if (vkContext.init(vkSetup) != VK_SUCCESS) {
      LOGE("Vulkan context creation failed\n"); return 1;
  }

  // Application window setup
  nvapp::ApplicationCreateInfo appInfo;
  appInfo.name           = "ETER — NURBS Viewer";
  appInfo.useMenu        = true;
  appInfo.instance       = vkContext.getInstance();
  appInfo.device         = vkContext.getDevice();
  appInfo.physicalDevice = vkContext.getPhysicalDevice();
  appInfo.queues         = vkContext.getQueueInfos();
  appInfo.dockSetup      = [](ImGuiID viewportID) {
      ImGuiID sid = ImGui::DockBuilderSplitNode(viewportID, ImGuiDir_Right, 0.25f, nullptr, &viewportID);
      ImGui::DockBuilderDockWindow("Settings", sid);
      ImGuiID lid = ImGui::DockBuilderSplitNode(viewportID, ImGuiDir_Down, 0.35f, nullptr, &viewportID);
      ImGui::DockBuilderDockWindow("Log", lid);
      ImGuiID pid = ImGui::DockBuilderSplitNode(lid, ImGuiDir_Right, 0.4f, nullptr, &lid);
      ImGui::DockBuilderDockWindow("Profiler", pid);
  };

  nvapp::Application app;
  app.init(appInfo);

  // Camera: must be added BEFORE nurbsRenderer so its matrices update before onRender
  app.addElement(std::make_shared<nvapp::ElementCamera>(nurbsRenderer->getCamera()));

  // Initialize camera position
  // Window size set in onAttach
  // Camera auto-fitted by bounding box

  app.addElement(nurbsRenderer);
  app.addElement(std::make_shared<nvapp::ElementProfiler>(&profilerManager));
  app.addElement(elementLogger);

  LOGI("%s", "Starting ETER NURBS viewer\n");
  app.run();           // main loop — runs until window closes
  app.deinit();        // cleanup Application (window, swapchain, ImGui)
  vkContext.deinit();  // cleanup Vulkan (device, instance)
  return 0;
}
