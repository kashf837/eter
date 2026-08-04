// ETER — NURBS real-time rendering
// Architecture: NURBS .surf → knot insertion → Bezier patches → GPU eval
//              → Vulkan triangle mesh → dynamic rendering → ImGui display
#define VMA_IMPLEMENTATION

#include <vector>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <backends/imgui_impl_vulkan.h>

#include "nurbs.h"  // nurbs_wireframe()
#include <nvapp/application.hpp>
#include <nvapp/elem_profiler.hpp>
#include <nvapp/elem_logger.hpp>
#include <nvapp/elem_default_menu.hpp>
#include <nvapp/elem_camera.hpp>
#include <nvutils/logger.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/context.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/default_structs.hpp>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/sampler_pool.hpp>
#include <nvvk/staging.hpp>
#include <nvvk/profiler_vk.hpp>
#include <nvutils/parameter_parser.hpp>

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
struct MVP { glm::mat4 model, view, proj; };

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
  NURBSRenderer(const Info& info) : m_info(info) {
    // Register command-line parameter: --surf <path> or -s <path>
    m_info.parameterRegistry->add({"surf"}, &m_surfPath);
  }

  // ---- Expose camera so ElementCamera can control it ----
  auto getCamera() { return m_camera; }

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

    // 5. Generate NURBS triangle mesh from .surf file
    createMeshBuffers(dev, m_surfPath.c_str());

    // 6. Register viewport image with ImGui (so onUIRender can display it)
    initViewportForImGui(dev);

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

    // Step C: draw the mesh
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

  void onPreRender() override { m_profilerTimeline->frameAdvance(); }

  // ============================================================
  // onUIMenu — menu bar (File / View)
  // ============================================================
  void onUIMenu() override
  {
    bool vsync = m_app->isVsync();
    if (ImGui::BeginMenu("File")) {
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
  // ---- Framework ----
  Info                          m_info;
  std::string                   m_surfPath = "data/wave.surf";  // can override via --surf
  nvapp::Application*           m_app{};
  nvvk::ResourceAllocator       m_alloc{};        // GPU memory manager (wraps VMA)
  nvvk::SamplerPool             m_samplerPool{};
  nvvk::StagingUploader         m_stagingUploader{}; // CPU→GPU data transfer helper
  nvutils::ProfilerTimeline*    m_profilerTimeline{};
  nvvk::ProfilerGpuTimer        m_profilerGpuTimer;

  // ---- Display images ----
  // Rendering pipeline:  mesh → m_renderTarget → blit → m_viewportImage → ImGui → screen
  int            m_imgW = 256, m_imgH = 256;
  nvvk::Image    m_viewportImage{};   // R32G32B32A32, ImGui display texture
  nvvk::Image    m_renderTarget{};    // R8G8B8A8, color attachment
  nvvk::Image    m_depthImage{};      // D32_SFLOAT, depth attachment (Z-buffer)
  VkDescriptorSet m_imguiImage{};
  bool           m_firstFrame = true;

  // ---- Camera ----
  std::shared_ptr<nvutils::CameraManipulator> m_camera = std::make_shared<nvutils::CameraManipulator>();

  // ---- Mesh geometry ----
  nvvk::Buffer m_vertexBuffer{};   // vertex positions (x,y,z per vertex)
  nvvk::Buffer m_indexBuffer{};    // triangle indices (3 x uint32_t per triangle)
  uint32_t     m_indexCount = 0;

  // ---- Graphics pipeline ----
  // Pipeline = shaders + rasterization state + blend state + depth state
  VkPipeline            m_pipeline       = VK_NULL_HANDLE;
  VkPipelineLayout      m_pipelineLayout = VK_NULL_HANDLE;
  VkDescriptorSetLayout m_dsLayout       = VK_NULL_HANDLE;
  VkDescriptorPool      m_dsPool         = VK_NULL_HANDLE;
  VkDescriptorSet       m_ds             = VK_NULL_HANDLE;
  nvvk::Buffer          m_ubo{};  // Uniform Buffer Object — holds MVP matrix for shader

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
    VkDescriptorSetLayoutBinding dslb = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT};
    VkDescriptorSetLayoutCreateInfo dslCI = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslCI.bindingCount = 1; dslCI.pBindings = &dslb;
    NVVK_CHECK(vkCreateDescriptorSetLayout(dev, &dslCI, nullptr, &m_dsLayout));

    VkPipelineLayoutCreateInfo plCI = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plCI.setLayoutCount = 1; plCI.pSetLayouts = &m_dsLayout;
    NVVK_CHECK(vkCreatePipelineLayout(dev, &plCI, nullptr, &m_pipelineLayout));

    // Vertex input: each vertex is 3 floats (x, y, z)
    VkVertexInputBindingDescription vib = {0, 3 * sizeof(float), VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription via = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0};
    VkPipelineVertexInputStateCreateInfo vi = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vi.vertexBindingDescriptionCount = 1; vi.pVertexBindingDescriptions = &vib;
    vi.vertexAttributeDescriptionCount = 1; vi.pVertexAttributeDescriptions = &via;

    // Standard pipeline state: triangle list, 1 viewport, no blending
    VkPipelineInputAssemblyStateCreateInfo ia = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo vp = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, 0, 0, 1, nullptr, 1, nullptr};
    VkPipelineRasterizationStateCreateInfo rs = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.lineWidth = 1.f;
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

  // Load NURBS surface from .surf → knot insertion → Bezier patches →
  // GPU evaluation → vertex & index buffers ready for rendering.
  void createMeshBuffers(VkDevice dev, const char* surfPath) {
    // Build camera params from current camera state
    CameraParams cam;
    glm::dvec3 eye = m_camera->getEye();
    cam.camX = (float)eye.x; cam.camY = (float)eye.y; cam.camZ = (float)eye.z;
    cam.fovY = 60.0f;
    cam.scrW = m_imgW; cam.scrH = m_imgH;

    // Adaptive mesh: per-patch tess factors based on camera distance + curvature
    uint32_t* idxData = nullptr;
    int totalVerts = 0, totalIndices = 0;
    float* verts = nurbs_adaptive_mesh(surfPath, &cam, &idxData, &totalVerts, &totalIndices);
    m_indexCount = totalIndices;
    if (!verts) { LOGE("Adaptive mesh failed: %s\n", surfPath); return; }

    // Upload vertex buffer
    VkDeviceSize vbSize = totalVerts * 3 * sizeof(float);
    m_alloc.createBuffer(m_vertexBuffer, vbSize,
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY);
    {
        void* ptr;
        NVVK_CHECK(m_stagingUploader.appendBufferMapping(m_vertexBuffer, 0, vbSize, ptr));
        memcpy(ptr, verts, vbSize);
    }

    // Upload index buffer
    VkDeviceSize ibSize = m_indexCount * sizeof(uint32_t);
    m_alloc.createBuffer(m_indexBuffer, ibSize,
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY);
    {
        void* ptr;
        NVVK_CHECK(m_stagingUploader.appendBufferMapping(m_indexBuffer, 0, ibSize, ptr));
        memcpy(ptr, idxData, ibSize);
    }

    // Submit all staging uploads at once
    VkCommandBuffer c = m_app->createTempCmdBuffer();
    m_stagingUploader.cmdUploadAppended(c);
    m_app->submitAndWaitTempCmdBuffer(c);
    m_stagingUploader.releaseStaging();
    free(verts); free(idxData);
    LOGI("Adaptive mesh: %d verts, %d indices (%d tris)\n",
         totalVerts, m_indexCount, m_indexCount / 3);
  }

  // Transition viewport image to readable layout, then register with ImGui.
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
    mvp.proj  = m_camera->getPerspectiveMatrix();
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
  nurbsRenderer->getCamera()->setWindowSize({256, 256});
  nurbsRenderer->getCamera()->setLookat({0.5f, 0.3f, 1.8f}, {0.5f, 0.5f, 0.16f}, {0, 1, 0});

  app.addElement(nurbsRenderer);
  app.addElement(std::make_shared<nvapp::ElementProfiler>(&profilerManager));
  app.addElement(elementLogger);

  LOGI("%s", "Starting ETER NURBS viewer\n");
  app.run();           // main loop — runs until window closes
  app.deinit();        // cleanup Application (window, swapchain, ImGui)
  vkContext.deinit();  // cleanup Vulkan (device, instance)
  return 0;
}
