// ETER — NURBS real-time rendering (ETER §6.1 Visibility Buffer)
// Architecture: NURBS .surf → knot insertion → Bezier patches → GPU eval
//   → Vulkan triangle mesh → [Pass 1: G-buffer] → [Pass 2: Resolve] → ImGui display
// G-buffer stores per-pixel (worldPos, worldNormal, patchID) for deferred lighting.
#define VMA_IMPLEMENTATION

#pragma region Includes
#include <vector>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <cctype>
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

// MVP matrices for visibility pass (set=0, binding=0 in visibility.vert)
struct MVP { glm::mat4 model, view, proj; };
// Lighting params for resolve pass (set=1, binding=0 in resolve.frag)
struct Lighting { float intensity, lightDirX, lightDirY, lightDirZ, checkerboard, boundary, patchCols, _pad; };
static_assert(sizeof(Lighting) == 32, "Lighting must be 32 bytes (2 vec4)");

// ============================================================
// Mesh — host-side triangle mesh data
// ============================================================
struct Mesh {
    std::vector<float>    v;  // interleaved xyz+nxnynz+uv+patchID (9 floats/vertex)
    std::vector<uint32_t> i;  // triangle indices
    std::vector<float>    cp; // Bézier CPs: 64 floats per patch (xyz+w × 16)
    float bbMin[3], bbMax[3];
    int   patchCount = 0;
    int   patchCols  = 0;  // nv: patches per row, for SSBO index decoding
    bool  empty() const { return i.empty(); }
    int   verts() const { return (int)(v.size()/9); }
    float scale() const { float dx=bbMax[0]-bbMin[0],dy=bbMax[1]-bbMin[1],dz=bbMax[2]-bbMin[2];
                           float s=dx>dy?dx:dy; s=s>dz?s:dz; return s<1.f?1.f:s; }
    float cx() const { return (bbMin[0]+bbMax[0])*0.5f; }
    float cy() const { return (bbMin[1]+bbMax[1])*0.5f; }
    float cz() const { return (bbMin[2]+bbMax[2])*0.5f; }
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
        float* bezierCPs = nullptr;
        float* verts = nurbs_adaptive_eval(path, cam, &nu, &nv, &info, &total, &bezierCPs);
        if (!verts) { fprintf(stderr, "  [SKIP] %s\n", path); return m; }
        m.patchCount = nu * nv;
        m.patchCols  = nv;
        if (bezierCPs) {
            int cpFloats = m.patchCount * 64;
            m.cp.assign(bezierCPs, bezierCPs + cpFloats);
            free(bezierCPs);
            // Center each patch's CPs for FP32 precision in SSBO eval.
            // Translating all CPs by the same offset cancels in du/dv
            // (derivative of constant = 0), so normals are unchanged.
            for (int p = 0; p < m.patchCount; p++) {
                float* cp = m.cp.data() + p * 64;
                float cx = 0, cy = 0, cz = 0;
                for (int k = 0; k < 16; k++) {
                    cx += cp[k*4+0]; cy += cp[k*4+1]; cz += cp[k*4+2];
                }
                cx /= 16.0f; cy /= 16.0f; cz /= 16.0f;
                for (int k = 0; k < 16; k++) {
                    cp[k*4+0] -= cx; cp[k*4+1] -= cy; cp[k*4+2] -= cz;
                }
            }
        }
        for (int pi = 0; pi < nu; pi++)
            for (int pj = 0; pj < nv; pj++) {
                PatchInfo* p = &info[pi*nv+pj];
                float* patch = verts + p->offset * 6;
                uint32_t base = (uint32_t)m.v.size() / 9;  // stride 9
                float patchID = (float)(pi * nv + pj);  // global sequential index
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
                        float u = p->tu > 1 ? (float)x / (float)(p->tu - 1) : 0.5f;
                        float v = p->tv > 1 ? (float)y / (float)(p->tv - 1) : 0.5f;
                        m.v.insert(m.v.end(),{px,py,pz,nx*inv,ny*inv,nz*inv,u,v,patchID});
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
        LOGI("Adaptive %s: %d verts, %d tris, %d patches\n", path, m.verts(), (int)(m.i.size()/3), m.patchCount);
        return m;
    }

    static Mesh single(const char* path, int res = 32) {
        Mesh m; resetBBox(m);
        int nu, nv; float* cp = nullptr;
        float* verts = nurbs_wireframe(path, res, &nu, &nv, &cp);
        if (!verts) { fprintf(stderr, "  [SKIP] %s\n", path); return m; }
        addPatchGrid(m, verts, cp, nu, nv, res);
        free(verts); free(cp);
        LOGI("Loaded %s: %d verts, %d tris\n", path, m.verts(), (int)(m.i.size()/3));
        return m;
    }

    static Mesh directory(const char* dirPath, int res = 32) {
        namespace fs = std::filesystem;
        Mesh m; resetBBox(m);
        std::vector<std::string> files;
        for (auto& e : fs::directory_iterator(dirPath))
            if (e.is_regular_file() && e.path().extension() == ".surf")
                files.push_back(e.path().string());
        std::sort(files.begin(), files.end());
        LOGI("Loading %zu surf files (res=%d)...\n", files.size(), res);
        if (files.empty()) {
            LOGI("WARNING: no .surf files in %s — STEP conversion may have failed\n", dirPath);
        }
        int skipped = 0;
        for (size_t i = 0; i < files.size(); i++) {
            if ((i+1) % 500 == 0) LOGI("  [%zu/%zu] ...\n", i+1, files.size());
            int nu, nv; float* cp = nullptr;
            float* verts = nurbs_wireframe(files[i].c_str(), res, &nu, &nv, &cp);
            if (!verts) { skipped++; free(cp); continue; }
            addPatchGrid(m, verts, cp, nu, nv, res);
            free(verts); free(cp);
        }
        LOGI("Done: %d verts, %d tris, %d skipped\n", m.verts(), (int)(m.i.size()/3), skipped);
        return m;
    }
private:

    static void resetBBox(Mesh& m) {
        m.bbMin[0]=m.bbMin[1]=m.bbMin[2]= 1e30f;
        m.bbMax[0]=m.bbMax[1]=m.bbMax[2]=-1e30f;
    }

    static void addPatchGrid(Mesh& m, const float* verts, const float* cp, int nu, int nv, int res) {
        m.patchCols = nv;  // for checkerboard decode in shader
        // Append CPs (pre-centering) for global indexing
        int cpBase = m.patchCount;
        if (cp) {
            m.cp.insert(m.cp.end(), cp, cp + nu * nv * 64);
        }
        for (int pi = 0; pi < nu; pi++)
            for (int pj = 0; pj < nv; pj++) {
                uint32_t base = (uint32_t)m.v.size() / 9;  // stride 9
                const float* p = verts + (pi*nv+pj)*res*res*6;
                float patchID = (float)(m.patchCount++);  // global sequential ID
                for (int y = 0; y < res; y++)
                    for (int x = 0; x < res; x++) {
                        int idx = y*res+x;
                        float px=p[idx*6+0], py=p[idx*6+1], pz=p[idx*6+2];
                        float gnx=p[idx*6+3], gny=p[idx*6+4], gnz=p[idx*6+5];
                        float gnLen = gnx*gnx+gny*gny+gnz*gnz;
                        if(gnLen < 1e-6f) {
                            int xR=(x+1<res)?x+1:x,xL=xR>0?xR-1:0, yD=(y+1<res)?y+1:y,yU=yD>0?yD-1:0;
                            float du[3]={p[(y*res+xR)*6+0]-p[(y*res+xL)*6+0],p[(y*res+xR)*6+1]-p[(y*res+xL)*6+1],p[(y*res+xR)*6+2]-p[(y*res+xL)*6+2]};
                            float dv[3]={p[(yD*res+x)*6+0]-p[(yU*res+x)*6+0],p[(yD*res+x)*6+1]-p[(yU*res+x)*6+1],p[(yD*res+x)*6+2]-p[(yU*res+x)*6+2]};
                            gnx=du[1]*dv[2]-du[2]*dv[1]; gny=du[2]*dv[0]-du[0]*dv[2]; gnz=du[0]*dv[1]-du[1]*dv[0];
                            gnLen = gnx*gnx+gny*gny+gnz*gnz;
                        }
                        float inv=1.f/(sqrtf(gnLen)+1e-10f);
                        float u = res > 1 ? (float)x / (float)(res - 1) : 0.5f;
                        float v = res > 1 ? (float)y / (float)(res - 1) : 0.5f;
                        m.v.insert(m.v.end(),{px,py,pz, gnx*inv,gny*inv,gnz*inv, u,v, patchID});
                        if(px<m.bbMin[0])m.bbMin[0]=px; if(py<m.bbMin[1])m.bbMin[1]=py; if(pz<m.bbMin[2])m.bbMin[2]=pz;
                        if(px>m.bbMax[0])m.bbMax[0]=px; if(py>m.bbMax[1])m.bbMax[1]=py; if(pz>m.bbMax[2])m.bbMax[2]=pz;
                    }
                for (int y = 0; y < res-1; y++)
                    for (int x = 0; x < res-1; x++) {
                        uint32_t a=base+y*res+x, b=a+1, c=a+res, d=c+1;
                        m.i.insert(m.i.end(),{a,b,c, b,d,c});
                    }
            }
        // Center CPs per patch for FP32 precision in SSBO eval
        if (cp) {
            for (int p = cpBase; p < m.patchCount; p++) {
                float* cpp = m.cp.data() + p * 64;
                float cx = 0, cy = 0, cz = 0;
                for (int k = 0; k < 16; k++) {
                    cx += cpp[k*4+0]; cy += cpp[k*4+1]; cz += cpp[k*4+2];
                }
                cx /= 16.0f; cy /= 16.0f; cz /= 16.0f;
                for (int k = 0; k < 16; k++) {
                    cpp[k*4+0] -= cx; cpp[k*4+1] -= cy; cpp[k*4+2] -= cz;
                }
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
    // If path is a directory on disk, override isDir regardless of what the
    // menu handler guessed — prevents trying to fopen a directory as a file.
    if (!isDir && std::filesystem::is_directory(path)) {
        LOGI("doReload: path is a directory, forcing isDir=true\n");
        isDir = true;
    }
    LOGI("doReload: %s (dir=%d)\n", path.c_str(), isDir);
    NVVK_CHECK(vkDeviceWaitIdle(m_app->getDevice()));
    if (m_indexCount > 0) {
        m_alloc.destroyBuffer(m_indexBuffer);
        m_alloc.destroyBuffer(m_vertexBuffer);
        m_indexCount = 0;
    }
    Mesh m;
    if (isDir) { m = MeshBuilder::directory(path.c_str(), 16); }  // lower res for multi-file STP
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
    createRenderTarget(dev);    // resolve pass output → blit source
    createDepthBuffer(dev);     // Z-buffer (shared by both passes)
    createGbuffer(dev);         // G-buffer: pos + normal + patchID

    // 2. Create Vulkan graphics pipelines
    createVisPipeline(dev);     // pass 1: mesh → G-buffer
    createResolvePipeline(dev); // pass 2: G-buffer → render target

    // 3. Create descriptor sets
    createVisDescriptorSet(dev);
    createResolveDescriptorSet(dev);

    // 4. Upload uniform buffers (MVP for vis pass, lighting for resolve pass)
    createUniformBuffers(dev);

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

    // Pipelines
    vkDestroyPipeline(dev, m_pipelineResolve, nullptr);
    vkDestroyPipeline(dev, m_pipelineVis, nullptr);
    vkDestroyPipelineLayout(dev, m_resolvePipelineLayout, nullptr);
    vkDestroyPipelineLayout(dev, m_visPipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(dev, m_resolveSsboLayout, nullptr);
    vkDestroyDescriptorSetLayout(dev, m_resolveDsLayout, nullptr);
    vkDestroyDescriptorSetLayout(dev, m_visDsLayout, nullptr);
    vkDestroyDescriptorPool(dev, m_resolveDsPool, nullptr);
    vkDestroyDescriptorPool(dev, m_visDsPool, nullptr);

    // Buffers & images
    for (int i = 0; i < MAX_FIF; i++) {
        vmaDestroyBuffer((VmaAllocator)m_alloc, m_lightUbo[i].buffer, m_lightUbo[i].allocation);
        vmaDestroyBuffer((VmaAllocator)m_alloc, m_visUbo[i].buffer, m_visUbo[i].allocation);
    }
    m_alloc.destroyBuffer(m_indexBuffer);
    m_alloc.destroyBuffer(m_vertexBuffer);
    m_alloc.destroyBuffer(m_bezierSSBO);  // always exists (dummy or real)
    m_alloc.destroyImage(m_gbufUV);
    m_alloc.destroyImage(m_gbufID);
    m_alloc.destroyImage(m_gbufNorm);
    m_alloc.destroyImage(m_gbufPos);
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
  // onRender — ETER §6.1 Visibility Buffer: two-pass rendering
  //   Pass 1 (visibility): mesh → G-buffer (pos + normal + patchID) + depth
  //   Pass 2 (resolve):     full-screen tri reads G-buffer → lighting → render target
  //   Then blit render target → viewport image for ImGui display.
  // ============================================================
  void onRender(VkCommandBuffer cmd) override
  {
    // Process pending mesh reload (deferred from menu callback).
    // Save+clear before processing: if doReload throws (e.g. OOM), we
    // don't retry every frame.
    if (!m_pendingLoad.empty()) {
        std::string loadPath = m_pendingLoad;
        bool        loadDir  = m_pendingIsDir;
        m_pendingLoad.clear();
        doReload(loadPath, loadDir);
        return;
    }
    updateUniforms();

    VkImageLayout undef = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout color = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    VkImageLayout depth = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    VkImageLayout ro    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkImageLayout src   = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    VkImageLayout dst   = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

    // --- First frame: transition all images from UNDEFINED to their initial layouts ---
    if (m_firstFrame) {
        m_firstFrame = false;
        // G-buffer images → COLOR_ATTACHMENT_OPTIMAL
        nvvk::cmdImageMemoryBarrier(cmd, {m_gbufPos.image,  undef, color});
        nvvk::cmdImageMemoryBarrier(cmd, {m_gbufNorm.image, undef, color});
        nvvk::cmdImageMemoryBarrier(cmd, {m_gbufID.image,   undef, color});
        nvvk::cmdImageMemoryBarrier(cmd, {m_gbufUV.image,   undef, color});
        // Depth → DEPTH_ATTACHMENT_OPTIMAL
        VkImageMemoryBarrier2 db = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        db.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        db.dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT;
        db.oldLayout = undef; db.newLayout = depth;
        db.image = m_depthImage.image;
        db.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
        VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &db;
        vkCmdPipelineBarrier2(cmd, &di);
        // Render target → COLOR_ATTACHMENT_OPTIMAL
        nvvk::cmdImageMemoryBarrier(cmd, {m_renderTarget.image, undef, color});
    }

    // ==============================================
    // PASS 1: Visibility — mesh → G-buffer + depth
    // ==============================================
    {
        VkRenderingAttachmentInfo gb[4] = {};
        // gbufPos: world-space position
        gb[0].sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        gb[0].imageView = m_gbufPos.descriptor.imageView;
        gb[0].imageLayout = color;
        gb[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        gb[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        gb[0].clearValue.color = {0.f, 0.f, 0.f, 0.f};
        // gbufNorm: world-space normal
        gb[1].sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        gb[1].imageView = m_gbufNorm.descriptor.imageView;
        gb[1].imageLayout = color;
        gb[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        gb[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        gb[1].clearValue.color = {0.f, 0.f, 0.f, 0.f};
        // gbufID: patch index
        gb[2].sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        gb[2].imageView = m_gbufID.descriptor.imageView;
        gb[2].imageLayout = color;
        gb[2].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        gb[2].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        gb[2].clearValue.color = {0.f, 0.f, 0.f, 0.f};
        // gbufUV: parametric (u,v)
        gb[3].sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        gb[3].imageView = m_gbufUV.descriptor.imageView;
        gb[3].imageLayout = color;
        gb[3].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        gb[3].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        gb[3].clearValue.color = {0.f, 0.f, 0.f, 0.f};

        VkRenderingAttachmentInfo da = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        da.imageView = m_depthImage.descriptor.imageView;
        da.imageLayout = depth;
        da.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        da.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        da.clearValue.depthStencil = {1.f, 0};

        VkRenderingInfo ri = {VK_STRUCTURE_TYPE_RENDERING_INFO};
        ri.renderArea = {{0, 0}, {(uint32_t)m_imgW, (uint32_t)m_imgH}};
        ri.layerCount = 1;
        ri.colorAttachmentCount = 4; ri.pColorAttachments = gb;
        ri.pDepthAttachment = &da;
        vkCmdBeginRendering(cmd, &ri);

        if (m_indexCount > 0) {
            VkViewport vp = {0, (float)m_imgH, (float)m_imgW, -(float)m_imgH, 0, 1};
            VkRect2D   sc = {{0, 0}, {(uint32_t)m_imgW, (uint32_t)m_imgH}};
            vkCmdSetViewport(cmd, 0, 1, &vp);
            vkCmdSetScissor(cmd, 0, 1, &sc);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineVis);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_visPipelineLayout, 0, 1, &m_visDs[m_frameIndex], 0, nullptr);
            VkDeviceSize off = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &m_vertexBuffer.buffer, &off);
            vkCmdBindIndexBuffer(cmd, m_indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(cmd, m_indexCount, 1, 0, 0, 0);
        }
        vkCmdEndRendering(cmd);
    }

    // --- Transition G-buffer: COLOR_ATTACHMENT → SHADER_READ_ONLY for resolve pass ---
    nvvk::cmdImageMemoryBarrier(cmd, {m_gbufPos.image,  color, ro});
    nvvk::cmdImageMemoryBarrier(cmd, {m_gbufNorm.image, color, ro});
    nvvk::cmdImageMemoryBarrier(cmd, {m_gbufID.image,   color, ro});
    nvvk::cmdImageMemoryBarrier(cmd, {m_gbufUV.image,   color, ro});

    // ==============================================
    // PASS 2: Resolve — full-screen tri → render target
    // ==============================================
    {
        VkRenderingAttachmentInfo ca = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        ca.imageView = m_renderTarget.descriptor.imageView;
        ca.imageLayout = color;
        ca.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        ca.clearValue.color = {0.1f, 0.1f, 0.15f, 1.f};

        VkRenderingInfo ri = {VK_STRUCTURE_TYPE_RENDERING_INFO};
        ri.renderArea = {{0, 0}, {(uint32_t)m_imgW, (uint32_t)m_imgH}};
        ri.layerCount = 1;
        ri.colorAttachmentCount = 1; ri.pColorAttachments = &ca;
        vkCmdBeginRendering(cmd, &ri);

        VkViewport vp = {0, (float)m_imgH, (float)m_imgW, -(float)m_imgH, 0, 1};
        VkRect2D   sc = {{0, 0}, {(uint32_t)m_imgW, (uint32_t)m_imgH}};
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &sc);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineResolve);
        VkDescriptorSet rds[3] = {m_resolveDs, m_resolveDs1[m_frameIndex], m_resolveDs2};
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_resolvePipelineLayout, 0, 3, rds, 0, nullptr);
        vkCmdDraw(cmd, 3, 1, 0, 0);  // full-screen triangle: 3 vertices, no vertex buffer

        vkCmdEndRendering(cmd);
    }

    // --- Transition G-buffer back to COLOR_ATTACHMENT for next frame's visibility pass ---
    nvvk::cmdImageMemoryBarrier(cmd, {m_gbufPos.image,  ro, color});
    nvvk::cmdImageMemoryBarrier(cmd, {m_gbufNorm.image, ro, color});
    nvvk::cmdImageMemoryBarrier(cmd, {m_gbufID.image,   ro, color});
    nvvk::cmdImageMemoryBarrier(cmd, {m_gbufUV.image,   ro, color});

    // ==============================================
    // Blit render target → viewport image for ImGui
    // ==============================================
    nvvk::cmdImageMemoryBarrier(cmd, {m_renderTarget.image, color, src});
    nvvk::cmdImageMemoryBarrier(cmd, {m_viewportImage.image, ro, dst});
    VkImageBlit blit = {};
    blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.srcOffsets[1]  = {m_imgW, m_imgH, 1};
    blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.dstOffsets[1]  = {m_imgW, m_imgH, 1};
    vkCmdBlitImage(cmd, m_renderTarget.image, src,
                   m_viewportImage.image, dst, 1, &blit, VK_FILTER_NEAREST);

    // Transition images back for next frame
    nvvk::cmdImageMemoryBarrier(cmd, {m_renderTarget.image, src, color});
    nvvk::cmdImageMemoryBarrier(cmd, {m_viewportImage.image, dst, ro});

    // Advance to next frame-in-flight slot after all rendering for this frame is recorded
    m_frameIndex = (m_frameIndex + 1) % MAX_FIF;
  }

  // ============================================================
  // onUIRender — ImGui panels
  // ============================================================
  void onUIRender() override
  {
    ImGui::Begin("Settings");
    ImGui::SliderFloat("Intensity", &m_intensity, 0.0f, 2.0f);
    ImGui::Checkbox("Checkerboard", &m_checkerboard);
    ImGui::Checkbox("Show Boundaries", &m_showBoundary);
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
    // Adaptive re-tessellation (ETER §4): only rebuild when camera IDLE.
    // Rebuilding during camera movement causes visible popping since each
    // frame's mesh has slightly different vertex positions.
    if (!m_lastPath.empty() && m_indexCount > 0) {
        glm::dvec3 e = m_camera->getEye();
        float dx=(float)e.x-m_lastEye[0], dy=(float)e.y-m_lastEye[1], dz=(float)e.z-m_lastEye[2];
        float dist = sqrtf(dx*dx+dy*dy+dz*dz);

        if (dist > m_modelScale * 0.15f) {
            // Camera moved enough: mark dirty, start/restart idle countdown
            m_rebuildPending = true;
            m_dirtyEye[0] = (float)e.x; m_dirtyEye[1] = (float)e.y; m_dirtyEye[2] = (float)e.z;
            m_idleFrames = 0;
        } else if (m_rebuildPending) {
            m_idleFrames++;
            if (m_idleFrames >= 30) {  // ~0.5s @60fps
                CameraParams cam;
                cam.camX = m_dirtyEye[0]; cam.camY = m_dirtyEye[1]; cam.camZ = m_dirtyEye[2];
                cam.fovY = 60.f; cam.scrW = m_imgW; cam.scrH = m_imgH;
                Mesh m = MeshBuilder::adaptive(m_lastPath.c_str(), &cam);
                if (!m.empty()) {
                    NVVK_CHECK(vkDeviceWaitIdle(m_app->getDevice()));
                    if (m_indexCount > 0) { m_alloc.destroyBuffer(m_indexBuffer); m_alloc.destroyBuffer(m_vertexBuffer); m_indexCount = 0; }
                    uploadMesh(m);
                    m_modelScale = m.scale();
                    m_center[0]=m.cx(); m_center[1]=m.cy(); m_center[2]=m.cz();
                    m_lastEye[0] = m_dirtyEye[0]; m_lastEye[1] = m_dirtyEye[1]; m_lastEye[2] = m_dirtyEye[2];
                    m_rebuildPending = false;
                    LOGI("[Adaptive] camera idle, %d verts\n", m.verts());
                }
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
                // Case-insensitive extension check
                std::string lower = sp;
                for (auto& c : lower) c = (char)tolower((unsigned char)c);
                if (lower.ends_with(".stp") || lower.ends_with(".step")) {
                    auto stem = path.stem().string();
                    m_tempDir = (path.parent_path() / (stem + "_parts")).string();
                    std::filesystem::create_directory(m_tempDir);
                    std::string cmd = "python scripts/step2surf.py --all \"" + sp + "\" \"" + m_tempDir + "/out\"";
                    LOGI("Converting STEP...\n");
                    int ret = system(cmd.c_str());
                    // Verify conversion succeeded and produced .surf files
                    bool hasFiles = false;
                    if (ret == 0) {
                        for (auto& e : std::filesystem::directory_iterator(m_tempDir))
                            if (e.is_regular_file() && e.path().extension() == ".surf") { hasFiles = true; break; }
                    }
                    if (hasFiles) {
                        m_pendingLoad = m_tempDir; m_pendingIsDir = true;
                    } else {
                        LOGI("STEP conversion failed (ret=%d, hasFiles=%d)\n", ret, (int)hasFiles);
                    }
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
  bool                          m_checkerboard    = false;
  bool                          m_showBoundary    = false;
  nvapp::Application*           m_app{};
  nvvk::ResourceAllocator       m_alloc{};        // GPU memory manager (wraps VMA)
  nvvk::SamplerPool             m_samplerPool{};
  nvvk::StagingUploader         m_stagingUploader{}; // CPU→GPU data transfer helper
  nvutils::ProfilerTimeline*    m_profilerTimeline{};
  nvvk::ProfilerGpuTimer        m_profilerGpuTimer;
#pragma endregion

#pragma region Display images
  // Rendering pipeline:  mesh → G-buffer (visPass) → resolve → m_renderTarget → blit → m_viewportImage → ImGui
  int            m_imgW = 1024, m_imgH = 1024;
  nvvk::Image    m_viewportImage{};   // R32G32B32A32, ImGui display texture
  nvvk::Image    m_renderTarget{};    // R8G8B8A8, resolve pass output → blit source
  nvvk::Image    m_depthImage{};      // D32_SFLOAT, depth attachment (Z-buffer, shared by both passes)
  nvvk::Image    m_gbufPos{};         // R32G32B32A32_SFLOAT: world-space position (xyz)
  nvvk::Image    m_gbufNorm{};        // R32G32B32A32_SFLOAT: world-space normal (xyz)
  nvvk::Image    m_gbufID{};          // R32_UINT: patch index
  nvvk::Image    m_gbufUV{};          // R32G32_SFLOAT: parametric (u,v) coordinates
  nvvk::Buffer   m_bezierSSBO{};      // SSBO: Bézier CPs (64 floats/patch, 16×vec4)
  bool           m_hasSSBO = false;   // true when SSBO is uploaded (adaptive path)
  int            m_patchCols = 1;     // nv: patches per row, for SSBO index decoding
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
  float m_dirtyEye[3] = {0,0,0};  // target eye for next rebuild
  int   m_idleFrames = 0;         // consecutive frames with negligible camera movement
  bool  m_rebuildPending = false; // true when waiting for camera to idle
#pragma endregion

#pragma region Graphics pipeline
  static constexpr int MAX_FIF = 3;  // frames in flight (triple-buffered UBOs)

  // Visibility pass pipeline (mesh → G-buffer)
  VkPipeline            m_pipelineVis       = VK_NULL_HANDLE;
  VkPipelineLayout      m_visPipelineLayout = VK_NULL_HANDLE;
  VkDescriptorSetLayout m_visDsLayout       = VK_NULL_HANDLE;
  VkDescriptorPool      m_visDsPool         = VK_NULL_HANDLE;
  VkDescriptorSet       m_visDs[MAX_FIF]    = {};
  nvvk::Buffer          m_visUbo[MAX_FIF]{};    // MVP UBO, one per frame-in-flight
  // Resolve pass pipeline (G-buffer → lighting)
  VkPipeline            m_pipelineResolve   = VK_NULL_HANDLE;
  VkPipelineLayout      m_resolvePipelineLayout = VK_NULL_HANDLE;
  VkDescriptorSetLayout m_resolveDsLayout   = VK_NULL_HANDLE;     // set=0: G-buffer images
  VkDescriptorSetLayout m_resolveSsboLayout = VK_NULL_HANDLE;     // set=2: Bézier CP SSBO
  VkDescriptorPool      m_resolveDsPool     = VK_NULL_HANDLE;
  VkDescriptorSet       m_resolveDs         = VK_NULL_HANDLE;     // set=0: G-buffer (shared)
  VkDescriptorSet       m_resolveDs1[MAX_FIF] = {};               // set=1: lighting UBO, per-frame
  VkDescriptorSet       m_resolveDs2        = VK_NULL_HANDLE;     // set=2: SSBO (shared)
  nvvk::Buffer          m_lightUbo[MAX_FIF]{};   // Lighting UBO, one per frame-in-flight
  int                   m_frameIndex = 0;         // cycles 0..MAX_FIF-1 each frame
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

  // G-buffer: three attachments for the visibility pass (ETER §6.1).
  //   gbufPos  — world-space position (R32G32B32A32_SFLOAT)
  //   gbufNorm — world-space normal   (R32G32B32A32_SFLOAT)
  //   gbufID   — patch index          (R32_UINT)
  // All must be SAMPLED so the resolve pass can read them as textures.
  void createGbuffer(VkDevice dev) {
    VkImageCreateInfo ii = DEFAULT_VkImageCreateInfo;
    ii.extent = {(uint32_t)m_imgW, (uint32_t)m_imgH, 1};
    ii.usage  = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    VkImageViewCreateInfo vi = DEFAULT_VkImageViewCreateInfo;

    ii.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    NVVK_CHECK(m_alloc.createImage(m_gbufPos, ii, vi));
    NVVK_CHECK(m_samplerPool.acquireSampler(m_gbufPos.descriptor.sampler));

    NVVK_CHECK(m_alloc.createImage(m_gbufNorm, ii, vi));
    NVVK_CHECK(m_samplerPool.acquireSampler(m_gbufNorm.descriptor.sampler));

    ii.format = VK_FORMAT_R32_UINT;
    NVVK_CHECK(m_alloc.createImage(m_gbufID, ii, vi));
    NVVK_CHECK(m_samplerPool.acquireSampler(m_gbufID.descriptor.sampler));

    // gbufUV: parametric (u,v) coordinates (R32G32_SFLOAT)
    ii.format = VK_FORMAT_R32G32_SFLOAT;
    NVVK_CHECK(m_alloc.createImage(m_gbufUV, ii, vi));
    NVVK_CHECK(m_samplerPool.acquireSampler(m_gbufUV.descriptor.sampler));
  }

  // Visibility pass pipeline (ETER §6.1 Pass 1): mesh → G-buffer.
  // Outputs worldPos, worldNormal, and patchID to 3 color attachments.
  void createVisPipeline(VkDevice dev) {

    // Load compiled SPIR-V binaries
    auto vCode = readFile("shaders/spv/visibility.vert.spv");
    auto fCode = readFile("shaders/spv/visibility.frag.spv");
    if (vCode.empty() || fCode.empty()) { LOGE("Shader load failed\n"); return; }

    // Create temporary shader modules (destroyed after pipeline creation)
    VkShaderModuleCreateInfo sm = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    sm.codeSize = vCode.size(); sm.pCode = (const uint32_t*)vCode.data();
    VkShaderModule vs; vkCreateShaderModule(dev, &sm, nullptr, &vs);
    sm.codeSize = fCode.size(); sm.pCode = (const uint32_t*)fCode.data();
    VkShaderModule fs; vkCreateShaderModule(dev, &sm, nullptr, &fs);

    // Descriptor set layout: just one UBO (MVP matrix) at set=0, binding=0.
    // Only the vertex shader uses it; fragment shader writes to G-buffer.
    VkDescriptorSetLayoutBinding dslb = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT};
    VkDescriptorSetLayoutCreateInfo dslCI = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslCI.bindingCount = 1; dslCI.pBindings = &dslb;
    NVVK_CHECK(vkCreateDescriptorSetLayout(dev, &dslCI, nullptr, &m_visDsLayout));

    VkPipelineLayoutCreateInfo plCI = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plCI.setLayoutCount = 1; plCI.pSetLayouts = &m_visDsLayout;
    NVVK_CHECK(vkCreatePipelineLayout(dev, &plCI, nullptr, &m_visPipelineLayout));

    // Vertex input: 9 floats per vertex (x,y,z + nx,ny,nz + u,v + patchID)
    VkVertexInputBindingDescription vib = {0, 9 * sizeof(float), VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription via[4] = {
        {0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0},           // location=0: position
        {1, 0, VK_FORMAT_R32G32B32_SFLOAT, 3*sizeof(float)}, // location=1: normal
        {2, 0, VK_FORMAT_R32G32_SFLOAT, 6*sizeof(float)}, // location=2: (u,v)
        {3, 0, VK_FORMAT_R32_SFLOAT,       8*sizeof(float)}, // location=3: patchID
    };
    VkPipelineVertexInputStateCreateInfo vi = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vi.vertexBindingDescriptionCount = 1; vi.pVertexBindingDescriptions = &vib;
    vi.vertexAttributeDescriptionCount = 4; vi.pVertexAttributeDescriptions = via;

    // Standard pipeline state: triangle list, 1 viewport, no blending
    VkPipelineInputAssemblyStateCreateInfo ia = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo vp = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, 0, 0, 1, nullptr, 1, nullptr};
    VkPipelineRasterizationStateCreateInfo rs = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.lineWidth = 1.f;
    VkPipelineMultisampleStateCreateInfo ms = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState cba[4] = {};
    for (int i = 0; i < 4; i++) cba[i].colorWriteMask = 0xF;
    VkPipelineColorBlendStateCreateInfo cb = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 4; cb.pAttachments = cba;

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

    // Dynamic rendering (Vulkan 1.3+): 4 color attachments (G-buffer) + depth
    VkFormat cf[4] = {VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_R32_UINT, VK_FORMAT_R32G32_SFLOAT};
    VkFormat df = VK_FORMAT_D32_SFLOAT;
    VkPipelineRenderingCreateInfo pr = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    pr.colorAttachmentCount = 4; pr.pColorAttachmentFormats = cf;
    pr.depthAttachmentFormat = df;

    VkGraphicsPipelineCreateInfo pi = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &pr};
    pi.stageCount = 2; pi.pStages = ss;
    pi.pVertexInputState = &vi; pi.pInputAssemblyState = &ia;
    pi.pViewportState = &vp; pi.pRasterizationState = &rs;
    pi.pMultisampleState = &ms; pi.pColorBlendState = &cb;
    pi.pDepthStencilState = &dss; pi.pDynamicState = &ds;
    pi.layout = m_visPipelineLayout;

    NVVK_CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &pi, nullptr, &m_pipelineVis));
    vkDestroyShaderModule(dev, vs, nullptr);
    vkDestroyShaderModule(dev, fs, nullptr);
    LOGI("Visibility pipeline created\n");
  }

  // Resolve pass pipeline (ETER §6.1 Pass 2): full-screen triangle → render target.
  // Reads G-buffer textures as combined image samplers, applies lighting.
  void createResolvePipeline(VkDevice dev) {
    auto vCode = readFile("shaders/spv/resolve.vert.spv");
    auto fCode = readFile("shaders/spv/resolve.frag.spv");
    if (vCode.empty() || fCode.empty()) { LOGE("Resolve shader load failed\n"); return; }

    VkShaderModuleCreateInfo sm = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    sm.codeSize = vCode.size(); sm.pCode = (const uint32_t*)vCode.data();
    VkShaderModule vs; vkCreateShaderModule(dev, &sm, nullptr, &vs);
    sm.codeSize = fCode.size(); sm.pCode = (const uint32_t*)fCode.data();
    VkShaderModule fs; vkCreateShaderModule(dev, &sm, nullptr, &fs);

    // Descriptor set layouts:
    //   set=0: 4 combined image samplers (gbufPos, gbufNorm, gbufID, gbufUV)
    //   set=1: 1 UBO (lighting params)
    //   set=2: 1 SSBO (Bézier CPs)
    VkDescriptorSetLayoutBinding bindings0[4] = {};
    for (int i = 0; i < 4; i++) {
        bindings0[i].binding = (uint32_t)i;
        bindings0[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings0[i].descriptorCount = 1;
        bindings0[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslCI0 = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslCI0.bindingCount = 4; dslCI0.pBindings = bindings0;

    VkDescriptorSetLayoutBinding binding1 = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT};
    VkDescriptorSetLayoutCreateInfo dslCI1 = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslCI1.bindingCount = 1; dslCI1.pBindings = &binding1;

    VkDescriptorSetLayoutBinding binding2 = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT};
    VkDescriptorSetLayoutCreateInfo dslCI2 = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslCI2.bindingCount = 1; dslCI2.pBindings = &binding2;

    VkDescriptorSetLayout layouts[3];
    NVVK_CHECK(vkCreateDescriptorSetLayout(dev, &dslCI0, nullptr, &layouts[0]));
    NVVK_CHECK(vkCreateDescriptorSetLayout(dev, &dslCI1, nullptr, &layouts[1]));
    NVVK_CHECK(vkCreateDescriptorSetLayout(dev, &dslCI2, nullptr, &layouts[2]));
    m_resolveDsLayout = layouts[0];    // save set=0 for pool creation ref
    m_resolveSsboLayout = layouts[2];  // save set=2 for pool creation ref

    VkPipelineLayoutCreateInfo plCI = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plCI.setLayoutCount = 3; plCI.pSetLayouts = layouts;
    NVVK_CHECK(vkCreatePipelineLayout(dev, &plCI, nullptr, &m_resolvePipelineLayout));

    // No vertex input — full-screen triangle uses gl_VertexIndex
    VkPipelineVertexInputStateCreateInfo vi = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vi.vertexBindingDescriptionCount = 0;
    vi.vertexAttributeDescriptionCount = 0;

    VkPipelineInputAssemblyStateCreateInfo ia = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vp = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, 0, 0, 1, nullptr, 1, nullptr};
    VkPipelineRasterizationStateCreateInfo rs = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.lineWidth = 1.f;

    VkPipelineMultisampleStateCreateInfo ms = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState cba = {}; cba.colorWriteMask = 0xF;
    VkPipelineColorBlendStateCreateInfo cb = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1; cb.pAttachments = &cba;

    // No depth test for full-screen quad
    VkPipelineDepthStencilStateCreateInfo dss = {VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    dss.depthTestEnable = VK_FALSE;
    dss.depthWriteEnable = VK_FALSE;

    VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo ds = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    ds.dynamicStateCount = 2; ds.pDynamicStates = dyn;

    VkPipelineShaderStageCreateInfo ss[] = {
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, 0, 0, VK_SHADER_STAGE_VERTEX_BIT, vs, "main"},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, 0, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fs, "main"},
    };

    VkFormat cf = VK_FORMAT_R8G8B8A8_UNORM;
    VkPipelineRenderingCreateInfo pr = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    pr.colorAttachmentCount = 1; pr.pColorAttachmentFormats = &cf;

    VkGraphicsPipelineCreateInfo pi = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &pr};
    pi.stageCount = 2; pi.pStages = ss;
    pi.pVertexInputState = &vi; pi.pInputAssemblyState = &ia;
    pi.pViewportState = &vp; pi.pRasterizationState = &rs;
    pi.pMultisampleState = &ms; pi.pColorBlendState = &cb;
    pi.pDepthStencilState = &dss; pi.pDynamicState = &ds;
    pi.layout = m_resolvePipelineLayout;

    NVVK_CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &pi, nullptr, &m_pipelineResolve));
    vkDestroyShaderModule(dev, vs, nullptr);
    vkDestroyShaderModule(dev, fs, nullptr);

    // Don't need set=1 layout reference after pipeline layout creation; save set=0 for pool
    // But we need to save set=1 layout reference too for pool creation. Use a local.
    vkDestroyDescriptorSetLayout(dev, layouts[1], nullptr);
    LOGI("Resolve pipeline created\n");
  }

  // Descriptor set for visibility pass: connects MVP UBO to the vertex shader.
  void createVisDescriptorSet(VkDevice dev) {
    VkDescriptorPoolSize dps = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, MAX_FIF};
    VkDescriptorPoolCreateInfo dpCI = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpCI.maxSets = MAX_FIF; dpCI.poolSizeCount = 1; dpCI.pPoolSizes = &dps;
    NVVK_CHECK(vkCreateDescriptorPool(dev, &dpCI, nullptr, &m_visDsPool));
    VkDescriptorSetLayout layouts[MAX_FIF];
    for (int i = 0; i < MAX_FIF; i++) layouts[i] = m_visDsLayout;
    VkDescriptorSetAllocateInfo dsa = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsa.descriptorPool = m_visDsPool; dsa.descriptorSetCount = MAX_FIF; dsa.pSetLayouts = layouts;
    NVVK_CHECK(vkAllocateDescriptorSets(dev, &dsa, m_visDs));
  }

  // Descriptor set for resolve pass: binds 3 G-buffer textures (set=0) + lighting UBO (set=1).
  // set=0 is shared across all frames (G-buffer images don't change).
  // set=1 is triple-buffered (one per frame-in-flight) for UBO coherency.
  void createResolveDescriptorSet(VkDevice dev) {
    int totalSets = 1 + MAX_FIF + 1;  // 1×set0 + MAX_FIF×set1 + 1×set2
    VkDescriptorPoolSize dps[3] = {
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4},          // gbufPos, Norm, ID, UV
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, (uint32_t)MAX_FIF},  // lighting UBO
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},                  // Bézier CP SSBO
    };
    VkDescriptorPoolCreateInfo dpCI = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpCI.maxSets = (uint32_t)totalSets; dpCI.poolSizeCount = 3; dpCI.pPoolSizes = dps;
    NVVK_CHECK(vkCreateDescriptorPool(dev, &dpCI, nullptr, &m_resolveDsPool));

    // Re-create temp set layouts for allocation (matching pipeline layout)
    VkDescriptorSetLayoutBinding bindings0[4] = {};
    for (int i = 0; i < 4; i++) {
        bindings0[i].binding = (uint32_t)i;
        bindings0[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings0[i].descriptorCount = 1;
        bindings0[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslCI0 = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslCI0.bindingCount = 4; dslCI0.pBindings = bindings0;
    VkDescriptorSetLayout set0Layout;
    NVVK_CHECK(vkCreateDescriptorSetLayout(dev, &dslCI0, nullptr, &set0Layout));

    VkDescriptorSetLayoutBinding binding1 = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT};
    VkDescriptorSetLayoutCreateInfo dslCI1 = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslCI1.bindingCount = 1; dslCI1.pBindings = &binding1;
    VkDescriptorSetLayout set1Layout;
    NVVK_CHECK(vkCreateDescriptorSetLayout(dev, &dslCI1, nullptr, &set1Layout));

    VkDescriptorSetLayout set1Layouts[MAX_FIF];
    for (int i = 0; i < MAX_FIF; i++) set1Layouts[i] = set1Layout;

    VkDescriptorSetAllocateInfo dsa = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsa.descriptorPool = m_resolveDsPool;
    // set=0: G-buffer textures (shared)
    dsa.descriptorSetCount = 1; dsa.pSetLayouts = &set0Layout;
    NVVK_CHECK(vkAllocateDescriptorSets(dev, &dsa, &m_resolveDs));
    // set=1: lighting UBO (per-frame)
    dsa.descriptorSetCount = MAX_FIF; dsa.pSetLayouts = set1Layouts;
    NVVK_CHECK(vkAllocateDescriptorSets(dev, &dsa, m_resolveDs1));
    // set=2: SSBO (shared)
    dsa.descriptorSetCount = 1; dsa.pSetLayouts = &m_resolveSsboLayout;
    NVVK_CHECK(vkAllocateDescriptorSets(dev, &dsa, &m_resolveDs2));

    // Write set=0: bind 4 G-buffer images as combined image samplers
    VkDescriptorImageInfo imgInfo[4] = {
        {m_gbufPos.descriptor.sampler,  m_gbufPos.descriptor.imageView,  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        {m_gbufNorm.descriptor.sampler, m_gbufNorm.descriptor.imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        {m_gbufID.descriptor.sampler,   m_gbufID.descriptor.imageView,   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        {m_gbufUV.descriptor.sampler,   m_gbufUV.descriptor.imageView,   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
    };
    VkWriteDescriptorSet wds[4] = {};
    for (int i = 0; i < 4; i++) {
        wds[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wds[i].dstSet = m_resolveDs;
        wds[i].dstBinding = (uint32_t)i;
        wds[i].descriptorCount = 1;
        wds[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        wds[i].pImageInfo = &imgInfo[i];
    }
    vkUpdateDescriptorSets(dev, 4, wds, 0, nullptr);

    vkDestroyDescriptorSetLayout(dev, set0Layout, nullptr);
    vkDestroyDescriptorSetLayout(dev, set1Layout, nullptr);
    LOGI("Resolve descriptor set created (%d FIF)\n", MAX_FIF);
  }

  // Uniform buffers: persistent-mapped CPU→GPU memory for per-frame data.
  // Triple-buffered: one copy per frame-in-flight so CPU never overwrites
  // UBO data that the GPU is still reading from a previous frame.
  void createUniformBuffers(VkDevice dev) {
    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    VmaAllocationCreateInfo aci = {};
    aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
              | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    aci.usage = VMA_MEMORY_USAGE_AUTO;
    VmaAllocationInfo ai;

    for (int i = 0; i < MAX_FIF; i++) {
        // Visibility UBO (MVP matrices)
        bci.size = sizeof(MVP);
        vmaCreateBuffer((VmaAllocator)m_alloc, &bci, &aci,
                        &m_visUbo[i].buffer, &m_visUbo[i].allocation, &ai);
        m_visUbo[i].mapping   = (uint8_t*)ai.pMappedData;
        m_visUbo[i].bufferSize = sizeof(MVP);

        VkDescriptorBufferInfo dbiVis = {m_visUbo[i].buffer, 0, sizeof(MVP)};
        VkWriteDescriptorSet wdsVis = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        wdsVis.dstSet = m_visDs[i]; wdsVis.dstBinding = 0; wdsVis.descriptorCount = 1;
        wdsVis.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; wdsVis.pBufferInfo = &dbiVis;

        // Lighting UBO (resolve pass)
        bci.size = sizeof(Lighting);
        vmaCreateBuffer((VmaAllocator)m_alloc, &bci, &aci,
                        &m_lightUbo[i].buffer, &m_lightUbo[i].allocation, &ai);
        m_lightUbo[i].mapping   = (uint8_t*)ai.pMappedData;
        m_lightUbo[i].bufferSize = sizeof(Lighting);

        VkDescriptorBufferInfo dbiLight = {m_lightUbo[i].buffer, 0, sizeof(Lighting)};
        VkWriteDescriptorSet wdsLight = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        wdsLight.dstSet = m_resolveDs1[i]; wdsLight.dstBinding = 0; wdsLight.descriptorCount = 1;
        wdsLight.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; wdsLight.pBufferInfo = &dbiLight;

        VkWriteDescriptorSet wds[2] = {wdsVis, wdsLight};
        vkUpdateDescriptorSets(dev, 2, wds, 0, nullptr);
    }

    // Dummy SSBO: 4 bytes, ensures descriptor set=2 is always valid.
    // Replaced with real Bézier CP data when adaptive mesh is loaded.
    VkBufferCreateInfo ssboCI = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    ssboCI.size = 4; ssboCI.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    VmaAllocationCreateInfo ssboACI = {};
    ssboACI.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    VmaAllocationInfo ssboAI;
    vmaCreateBuffer((VmaAllocator)m_alloc, &ssboCI, &ssboACI,
                    &m_bezierSSBO.buffer, &m_bezierSSBO.allocation, &ssboAI);
    m_bezierSSBO.bufferSize = 4;
    m_hasSSBO = false;

    VkDescriptorBufferInfo dbiSSBO = {m_bezierSSBO.buffer, 0, 4};
    VkWriteDescriptorSet wdsSSBO = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    wdsSSBO.dstSet = m_resolveDs2; wdsSSBO.dstBinding = 0; wdsSSBO.descriptorCount = 1;
    wdsSSBO.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wdsSSBO.pBufferInfo = &dbiSSBO;
    vkUpdateDescriptorSets(dev, 1, &wdsSSBO, 0, nullptr);
  }

  void uploadMesh(const Mesh& m) {
    VkDevice dev = m_app->getDevice();
    VkDeviceSize vsz = m.v.size() * sizeof(float);
    m_alloc.createBuffer(m_vertexBuffer, vsz, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
    { void* p; NVVK_CHECK(m_stagingUploader.appendBufferMapping(m_vertexBuffer, 0, vsz, p)); memcpy(p, m.v.data(), vsz); }
    VkDeviceSize isz = m.i.size() * sizeof(uint32_t);
    m_alloc.createBuffer(m_indexBuffer, isz, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
    { void* p; NVVK_CHECK(m_stagingUploader.appendBufferMapping(m_indexBuffer, 0, isz, p)); memcpy(p, m.i.data(), isz); }

    // Upload Bézier CP SSBO if available (adaptive path with ETER §6.2)
    if (!m.cp.empty() && m.patchCount > 0) {
        VkDeviceSize cpsz = m.cp.size() * sizeof(float);
        m_alloc.destroyBuffer(m_bezierSSBO);  // always exists (dummy or real)
        m_alloc.createBuffer(m_bezierSSBO, cpsz,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
        { void* p; NVVK_CHECK(m_stagingUploader.appendBufferMapping(m_bezierSSBO, 0, cpsz, p)); memcpy(p, m.cp.data(), cpsz); }
        m_hasSSBO = true;

        // Write SSBO descriptor (if descriptor set is already created)
        if (m_resolveDs2 != VK_NULL_HANDLE) {
            VkDescriptorBufferInfo dbiSSBO = {m_bezierSSBO.buffer, 0, cpsz};
            VkWriteDescriptorSet wdsSSBO = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            wdsSSBO.dstSet = m_resolveDs2; wdsSSBO.dstBinding = 0; wdsSSBO.descriptorCount = 1;
            wdsSSBO.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wdsSSBO.pBufferInfo = &dbiSSBO;
            vkUpdateDescriptorSets(dev, 1, &wdsSSBO, 0, nullptr);
        }
    } else {
        m_hasSSBO = false;
    }

    VkCommandBuffer c = m_app->createTempCmdBuffer();
    m_stagingUploader.cmdUploadAppended(c);
    m_app->submitAndWaitTempCmdBuffer(c);
    m_stagingUploader.releaseStaging();
    m_indexCount = (uint32_t)m.i.size();
    m_patchCols  = m.patchCols > 0 ? m.patchCols : 1;
    LOGI("Uploaded: %d verts, %d tris, %d CPs\n", m.verts(), (int)(m.i.size()/3), (int)m.cp.size());
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
  void updateUniforms() {
    // MVP for visibility pass
    MVP mvp;
    mvp.model = glm::mat4(1.f);
    mvp.view  = m_camera->getViewMatrix();
    auto vsz  = m_app->getViewportSize();
    float asp = vsz.width / (float)vsz.height;
    float nearP = m_modelScale * 0.001f;
    float farP  = m_modelScale * 5.0f;
    if (nearP < 0.001f) nearP = 0.001f;
    mvp.proj  = glm::perspective(glm::radians(60.f), asp, nearP, farP);
    mvp.proj[1][1] *= -1.f; // Vulkan Y flip
    memcpy(m_visUbo[m_frameIndex].mapping, &mvp, sizeof(MVP));

    // Lighting for resolve pass — light comes from camera (headlight model)
    Lighting light = {};
    light.intensity    = m_intensity;
    light.checkerboard = m_checkerboard ? 1.0f : 0.0f;
    light.boundary     = m_showBoundary   ? 1.0f : 0.0f;
    light.patchCols    = (float)m_patchCols;
    // Camera-view lighting: direction from model center toward camera eye
    glm::dvec3 eye = m_camera->getEye();
    glm::vec3 lightDir = glm::normalize(glm::vec3(eye.x - m_center[0],
                                                   eye.y - m_center[1],
                                                   eye.z - m_center[2]));
    light.lightDirX = lightDir.x;
    light.lightDirY = lightDir.y;
    light.lightDirZ = lightDir.z;
    memcpy(m_lightUbo[m_frameIndex].mapping, &light, sizeof(Lighting));
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
  vkSetup.deviceExtensions   = {{VK_KHR_SWAPCHAIN_EXTENSION_NAME}};
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
