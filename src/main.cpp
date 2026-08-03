/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// ETER — NURBS real-time rendering (WIP)
#define VMA_IMPLEMENTATION

#include <vector>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <backends/imgui_impl_vulkan.h>

#include "nurbs.h"
#include <nvapp/application.hpp>
#include <nvapp/elem_profiler.hpp>
#include <nvapp/elem_logger.hpp>
#include <nvapp/elem_default_menu.hpp>
#include <nvapp/elem_default_title.hpp>
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

static std::vector<char> readFile(const std::string& fn) {
    std::ifstream f(fn, std::ios::ate | std::ios::binary);
    if (!f) return {};
    size_t sz = f.tellg(); std::vector<char> buf(sz);
    f.seekg(0); f.read(buf.data(), sz);
    return buf;
}

struct MVP { glm::mat4 model, view, proj; };

class SampleElement : public nvapp::IAppElement
{
public:
  struct Info { nvutils::ProfilerManager* profilerManager{}; nvutils::ParameterRegistry* parameterRegistry{}; };
  SampleElement(const Info& info) : m_info(info) { }
  auto getCamera() { return m_camera; }

  std::shared_ptr<nvutils::CameraManipulator> m_camera = std::make_shared<nvutils::CameraManipulator>();

  void onAttach(nvapp::Application* app) override
  {
    m_app = app;
    m_camera->setWindowSize({m_imgW, m_imgH});
    m_camera->setLookat({0.5f, 0.3f, 1.8f}, {0.5f, 0.5f, 0.16f}, {0, 1, 0});

    VmaAllocatorCreateInfo ai = {};
    ai.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    ai.physicalDevice = app->getPhysicalDevice();
    ai.device = app->getDevice();
    ai.instance = app->getInstance();
    NVVK_CHECK(m_alloc.init(ai));
    m_samplerPool.init(app->getDevice());
    m_stagingUploader.init(&m_alloc, true);
    VkDevice dev = app->getDevice();

    LOGI("onAttach start\n");
    // ---- 1. Viewport image (for ImGui display) ----
    {
        VkImageCreateInfo ii = DEFAULT_VkImageCreateInfo;
        ii.extent = {256,256,1}; ii.format = VK_FORMAT_R32G32B32A32_SFLOAT;
        ii.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        VkImageViewCreateInfo vi = DEFAULT_VkImageViewCreateInfo;
        vi.components = {.a = VK_COMPONENT_SWIZZLE_ONE};
        NVVK_CHECK(m_alloc.createImage(m_viewportImage, ii, vi));
        NVVK_CHECK(m_samplerPool.acquireSampler(m_viewportImage.descriptor.sampler));
    }

    // ---- 2. Render target (R8G8B8A8, supports color attachment) ----
    {
        VkImageCreateInfo ii = DEFAULT_VkImageCreateInfo;
        ii.extent = {256,256,1}; ii.format = VK_FORMAT_R8G8B8A8_UNORM;
        ii.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        VkImageViewCreateInfo vi = DEFAULT_VkImageViewCreateInfo;
        NVVK_CHECK(m_alloc.createImage(m_renderTarget, ii, vi));
    }

    // ---- 2b. Depth image ----
    {
        VkImageCreateInfo ii = DEFAULT_VkImageCreateInfo;
        ii.extent = {256,256,1}; ii.format = VK_FORMAT_D32_SFLOAT;
        ii.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        VkImageViewCreateInfo vi = DEFAULT_VkImageViewCreateInfo;
        vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        NVVK_CHECK(m_alloc.createImage(m_depthImage, ii, vi));
    }

    // ---- 3. Shaders & pipeline ----
    auto vCode = readFile("shaders/spv/mesh.vert.spv");
    auto fCode = readFile("shaders/spv/mesh.frag.spv");
    LOGI("Shader sizes: vert=%zu frag=%zu\n", vCode.size(), fCode.size());
    if (vCode.empty()||fCode.empty()) { LOGE("Shader load failed\n"); return; }

    VkShaderModuleCreateInfo sm = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    sm.codeSize=vCode.size(); sm.pCode=(const uint32_t*)vCode.data();
    VkShaderModule vs; vkCreateShaderModule(dev,&sm,nullptr,&vs);
    sm.codeSize=fCode.size(); sm.pCode=(const uint32_t*)fCode.data();
    VkShaderModule fs; vkCreateShaderModule(dev,&sm,nullptr,&fs);

    // Descriptor set layout (UBO at set=0 binding=0)
    VkDescriptorSetLayoutBinding dslb = {0,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1,VK_SHADER_STAGE_VERTEX_BIT};
    VkDescriptorSetLayoutCreateInfo dslCI = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslCI.bindingCount=1; dslCI.pBindings=&dslb;
    NVVK_CHECK(vkCreateDescriptorSetLayout(dev,&dslCI,nullptr,&m_dsLayout));

    // Pipeline layout
    VkPipelineLayoutCreateInfo plCI = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plCI.setLayoutCount=1; plCI.pSetLayouts=&m_dsLayout;
    NVVK_CHECK(vkCreatePipelineLayout(dev,&plCI,nullptr,&m_pipelineLayout));

    // Vertex input
    VkVertexInputBindingDescription vib = {0,3*sizeof(float),VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription via = {0,0,VK_FORMAT_R32G32B32_SFLOAT,0};
    VkPipelineVertexInputStateCreateInfo vi = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vi.vertexBindingDescriptionCount=1; vi.pVertexBindingDescriptions=&vib;
    vi.vertexAttributeDescriptionCount=1; vi.pVertexAttributeDescriptions=&via;

    VkPipelineInputAssemblyStateCreateInfo ia = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo vp = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO}; vp.viewportCount=1; vp.scissorCount=1;
    VkPipelineRasterizationStateCreateInfo rs = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO}; rs.lineWidth=1.f;
    VkPipelineMultisampleStateCreateInfo ms = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO}; ms.rasterizationSamples=VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendAttachmentState cba = {}; cba.colorWriteMask=0xF;
    VkPipelineColorBlendStateCreateInfo cb = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO}; cb.attachmentCount=1; cb.pAttachments=&cba;
    VkPipelineDepthStencilStateCreateInfo dss = {VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    dss.depthTestEnable=VK_TRUE; dss.depthWriteEnable=VK_TRUE;
    dss.depthCompareOp=VK_COMPARE_OP_LESS_OR_EQUAL;
    VkDynamicState dyn[]={VK_DYNAMIC_STATE_VIEWPORT,VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo ds = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO}; ds.dynamicStateCount=2; ds.pDynamicStates=dyn;
    VkPipelineShaderStageCreateInfo ss[]={
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,0,0,VK_SHADER_STAGE_VERTEX_BIT,vs,"main"},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,0,0,VK_SHADER_STAGE_FRAGMENT_BIT,fs,"main"},
    };
    VkFormat cf = VK_FORMAT_R8G8B8A8_UNORM;
    VkFormat df = VK_FORMAT_D32_SFLOAT;
    VkPipelineRenderingCreateInfo pr = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    pr.colorAttachmentCount=1; pr.pColorAttachmentFormats=&cf;
    pr.depthAttachmentFormat=df;
    VkGraphicsPipelineCreateInfo pi = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO}; pi.pNext=&pr;
    pi.stageCount=2; pi.pStages=ss; pi.pVertexInputState=&vi; pi.pInputAssemblyState=&ia;
    pi.pViewportState=&vp; pi.pRasterizationState=&rs; pi.pMultisampleState=&ms;
    pi.pColorBlendState=&cb; pi.pDepthStencilState=&dss;
    pi.pDynamicState=&ds; pi.layout=m_pipelineLayout;
    NVVK_CHECK(vkCreateGraphicsPipelines(dev,VK_NULL_HANDLE,1,&pi,nullptr,&m_pipeline));
    LOGI("Pipeline created\n");
    vkDestroyShaderModule(dev,vs,nullptr); vkDestroyShaderModule(dev,fs,nullptr);

    // ---- 4. Descriptor pool & set ----
    LOGI("Creating descriptor pool...\n");
    VkDescriptorPoolSize dps = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1};
    VkDescriptorPoolCreateInfo dpCI = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; dpCI.maxSets=1; dpCI.poolSizeCount=1; dpCI.pPoolSizes=&dps;
    NVVK_CHECK(vkCreateDescriptorPool(dev,&dpCI,nullptr,&m_dsPool));
    VkDescriptorSetAllocateInfo dsa = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; dsa.descriptorPool=m_dsPool; dsa.descriptorSetCount=1; dsa.pSetLayouts=&m_dsLayout;
    NVVK_CHECK(vkAllocateDescriptorSets(dev,&dsa,&m_ds));
    LOGI("Descriptor done\n");

    // ---- 5. Uniform buffer (MVP) — host-visible for per-frame camera updates ----
    {
        VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = sizeof(MVP); bci.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        VmaAllocationCreateInfo aci = {};
        aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        aci.usage = VMA_MEMORY_USAGE_AUTO;
        VmaAllocationInfo ai;
        vmaCreateBuffer((VmaAllocator)m_alloc, &bci, &aci, &m_ubo.buffer, &m_ubo.allocation, &ai);
        m_ubo.mapping = (uint8_t*)ai.pMappedData;
        m_ubo.bufferSize = sizeof(MVP);
        // Init with identity
        MVP mvp;
        mvp.model = glm::mat4(1.f);
        mvp.view  = glm::lookAt(glm::vec3(1.2f,1.5f,2.f), glm::vec3(0.5f,0.5f,0.16f), glm::vec3(0,1,0));
        mvp.proj  = glm::perspective(glm::radians(60.f), 1.f, 0.1f, 10.f);
        mvp.proj[1][1] *= -1.f;
        memcpy(m_ubo.mapping, &mvp, sizeof(MVP));
    }
    VkDescriptorBufferInfo dbi = {m_ubo.buffer,0,sizeof(MVP)};
    VkWriteDescriptorSet wds = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; wds.dstSet=m_ds; wds.dstBinding=0; wds.descriptorCount=1; wds.descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; wds.pBufferInfo=&dbi;
    vkUpdateDescriptorSets(dev,1,&wds,0,nullptr);
    LOGI("UBO done\n");

    // ---- 6. NURBS vertex & index buffers ----
    {
        int np_u, np_v, res = 12;
        float* verts = nurbs_wireframe("data/wave.surf", res, &np_u, &np_v);
        if (!verts) { LOGE("Wireframe failed\n"); return; }
        int totalVerts = np_u * np_v * res * res;
        VkDeviceSize vbSize = totalVerts * 3 * sizeof(float);

        // Vertex buffer via staging uploader
        m_alloc.createBuffer(m_vertexBuffer, vbSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        { void* ptr; NVVK_CHECK(m_stagingUploader.appendBufferMapping(m_vertexBuffer, 0, vbSize, ptr));
          float* dst = (float*)ptr;
          for (int pi=0; pi<np_u; pi++) for (int pj=0; pj<np_v; pj++) {
              float* v = verts + (pi*np_v+pj)*res*res*3;
              for (int y=0; y<res; y++) for (int x=0; x<res; x++) {
                  *dst++ = v[(y*res+x)*3+0];
                  *dst++ = v[(y*res+x)*3+1];
                  *dst++ = v[(y*res+x)*3+2];
              }
          }
        }

        // Index buffer via staging uploader
        int quadsPerPatch = (res-1)*(res-1);
        m_indexCount = np_u * np_v * quadsPerPatch * 6;
        VkDeviceSize ibSize = m_indexCount * sizeof(uint32_t);
        m_alloc.createBuffer(m_indexBuffer, ibSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        { void* ptr; NVVK_CHECK(m_stagingUploader.appendBufferMapping(m_indexBuffer, 0, ibSize, ptr));
          uint32_t* idx = (uint32_t*)ptr;
          for (int pi=0; pi<np_u; pi++) for (int pj=0; pj<np_v; pj++) {
              uint32_t base = (pi*np_v + pj) * res * res;
              for (int y=0; y<res-1; y++) for (int x=0; x<res-1; x++) {
                  uint32_t a=base+y*res+x, b=a+1, c=a+res, d=c+1;
                  *idx++=a;*idx++=b;*idx++=c;
                  *idx++=b;*idx++=d;*idx++=c;
              }
          }
        }
        free(verts);

        // Submit all staging uploads together
        VkCommandBuffer c = m_app->createTempCmdBuffer();
        m_stagingUploader.cmdUploadAppended(c);
        m_app->submitAndWaitTempCmdBuffer(c);
        m_stagingUploader.releaseStaging();
        LOGI("Verts=%d Indices=%d\n", totalVerts, m_indexCount);
    }
    LOGI("Buffers created\n");

    // Transition viewport image UNDEFINED → SHADER_READ_ONLY for ImGui
    {
        VkCommandBuffer c = m_app->createTempCmdBuffer();
        nvvk::cmdImageMemoryBarrier(c, {m_viewportImage.image,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL});
        m_app->submitAndWaitTempCmdBuffer(c);
    }

    // ---- 7. ImGui display texture ----
    m_imguiImage = ImGui_ImplVulkan_AddTexture(m_viewportImage.descriptor.sampler,
        m_viewportImage.descriptor.imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    m_profilerTimeline = m_info.profilerManager->createTimeline({"graphics"});
    m_profilerGpuTimer.init(m_profilerTimeline, dev, app->getPhysicalDevice(), app->getQueue(0).familyIndex, true);
    LOGI("Pipeline ready, onAttach done\n");
  }

  void onDetach() override
  {
    NVVK_CHECK(vkDeviceWaitIdle(m_app->getDevice()));
    VkDevice dev = m_app->getDevice();
    ImGui_ImplVulkan_RemoveTexture(m_imguiImage);
    vkDestroyPipeline(dev, m_pipeline, nullptr);
    vkDestroyPipelineLayout(dev, m_pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(dev, m_dsLayout, nullptr);
    vkDestroyDescriptorPool(dev, m_dsPool, nullptr);
    m_alloc.destroyBuffer(m_ubo);
    m_alloc.destroyBuffer(m_indexBuffer);
    m_alloc.destroyBuffer(m_vertexBuffer);
    m_alloc.destroyImage(m_renderTarget);
    m_alloc.destroyImage(m_depthImage);
    m_alloc.destroyImage(m_viewportImage);
    m_stagingUploader.deinit();
    m_samplerPool.deinit();
    m_alloc.deinit();
    m_profilerGpuTimer.deinit();
    m_info.profilerManager->destroyTimeline(m_profilerTimeline);
  }

  void onUIRender() override
  {
    ImGui::Begin("Settings");
    ImGui::Text("3D Mesh — %d triangles", m_indexCount/3);
    ImGui::TextDisabled("%d FPS / %.3fms", static_cast<int>(ImGui::GetIO().Framerate), 1000.F / ImGui::GetIO().Framerate);

    // Add window information
    const VkExtent2D& viewportSize = m_app->getViewportSize();
    ImGui::Text("Viewport Size: %d x %d", viewportSize.width, viewportSize.height);

    ImGui::End();

    // Rendered image displayed fully in 'Viewport' window
    ImGui::Begin("Viewport");
    ImGui::Image((ImTextureID)m_imguiImage, ImGui::GetContentRegionAvail());
    ImGui::End();
  }

  void onPreRender() override { m_profilerTimeline->frameAdvance(); }

  void onRender(VkCommandBuffer cmd)
  {
    // Update camera matrices in UBO
    MVP mvp;
    mvp.model = glm::mat4(1.f);
    mvp.view  = m_camera->getViewMatrix();
    mvp.proj  = m_camera->getPerspectiveMatrix();
    memcpy(m_ubo.mapping, &mvp, sizeof(MVP));

    // 1. Transition render target to COLOR_ATTACHMENT
    VkImageLayout rtOld = m_firstFrame ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    m_firstFrame = false;
    if (rtOld == VK_IMAGE_LAYOUT_UNDEFINED) {
        nvvk::cmdImageMemoryBarrier(cmd, {m_renderTarget.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});
        // Depth barrier with explicit aspect
        VkImageMemoryBarrier2 db = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        db.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        db.dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT;
        db.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; db.newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        db.image = m_depthImage.image;
        db.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
        VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &db;
        vkCmdPipelineBarrier2(cmd, &di);
    }

    // 2. Dynamic rendering on render target + depth
    VkRenderingAttachmentInfo da = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    da.imageView=m_depthImage.descriptor.imageView;
    da.imageLayout=VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    da.loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR; da.storeOp=VK_ATTACHMENT_STORE_OP_STORE;
    da.clearValue.depthStencil = {1.f, 0};

    VkRenderingAttachmentInfo ca = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    ca.imageView=m_renderTarget.descriptor.imageView;
    ca.imageLayout=VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    ca.loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR; ca.storeOp=VK_ATTACHMENT_STORE_OP_STORE;
    ca.clearValue.color = {0.1f,0.1f,0.15f,1.f};

    VkRenderingInfo ri = {VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea={{0,0},{256,256}}; ri.layerCount=1;
    ri.colorAttachmentCount=1; ri.pColorAttachments=&ca;
    ri.pDepthAttachment=&da;
    vkCmdBeginRendering(cmd, &ri);

    VkViewport vp = {0,256,256,-256,0,1}; // Vulkan Y flip
    VkRect2D sc = {{0,0},{256,256}};
    vkCmdSetViewport(cmd,0,1,&vp); vkCmdSetScissor(cmd,0,1,&sc);
    vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,m_pipeline);
    vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,m_pipelineLayout,0,1,&m_ds,0,nullptr);
    VkDeviceSize off=0;
    vkCmdBindVertexBuffers(cmd,0,1,&m_vertexBuffer.buffer,&off);
    vkCmdBindIndexBuffer(cmd,m_indexBuffer.buffer,0,VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cmd,m_indexCount,1,0,0,0);
    vkCmdEndRendering(cmd);

    // 3. Blit renderTarget → viewportImage
    nvvk::cmdImageMemoryBarrier(cmd, {m_renderTarget.image,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL});
    nvvk::cmdImageMemoryBarrier(cmd, {m_viewportImage.image,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL});
    VkImageBlit blit = {};
    blit.srcSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1}; blit.srcOffsets[1]={256,256,1};
    blit.dstSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1}; blit.dstOffsets[1]={256,256,1};
    vkCmdBlitImage(cmd, m_renderTarget.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   m_viewportImage.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_NEAREST);

    // 4. Transition back for next frame
    nvvk::cmdImageMemoryBarrier(cmd, {m_renderTarget.image,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});
    nvvk::cmdImageMemoryBarrier(cmd, {m_viewportImage.image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL});
  }

  // Called if showMenu is true
  void onUIMenu() override
  {
    bool vsync = m_app->isVsync();

    if(ImGui::BeginMenu("File"))
    {
      if(ImGui::MenuItem("Exit", "Ctrl+Q"))
        m_app->close();
      ImGui::EndMenu();
    }
    if(ImGui::BeginMenu("View"))
    {
      ImGui::MenuItem("V-Sync", "Ctrl+Shift+V", &vsync);
      ImGui::EndMenu();
    }

    if(ImGui::IsKeyPressed(ImGuiKey_Q) && ImGui::IsKeyDown(ImGuiKey_LeftCtrl))
    {
      m_app->close();
    }

    if(ImGui::IsKeyPressed(ImGuiKey_V) && ImGui::IsKeyDown(ImGuiKey_LeftCtrl) && ImGui::IsKeyDown(ImGuiKey_LeftShift))
    {
      vsync = !vsync;
    }

    if(vsync != m_app->isVsync())
    {
      m_app->setVsync(vsync);
    }
  }

private:
  Info m_info;
  int  m_imgW = 256, m_imgH = 256;

  nvvk::ResourceAllocator m_alloc{};
  nvapp::Application*     m_app{};
  nvvk::SamplerPool       m_samplerPool{};
  nvvk::StagingUploader   m_stagingUploader{};
  nvutils::ProfilerTimeline* m_profilerTimeline{};
  nvvk::ProfilerGpuTimer     m_profilerGpuTimer;

  // Display
  nvvk::Image       m_viewportImage{};
  nvvk::Image       m_renderTarget{};
  nvvk::Image       m_depthImage{};
  VkDescriptorSet   m_imguiImage{};
  bool m_firstFrame = true;

  // Geometry
  nvvk::Buffer m_vertexBuffer{};
  nvvk::Buffer m_indexBuffer{};
  uint32_t     m_indexCount = 0;

  // Pipeline
  VkPipeline            m_pipeline       = VK_NULL_HANDLE;
  VkPipelineLayout      m_pipelineLayout = VK_NULL_HANDLE;
  VkDescriptorSetLayout m_dsLayout       = VK_NULL_HANDLE;
  VkDescriptorPool      m_dsPool         = VK_NULL_HANDLE;
  VkDescriptorSet       m_ds             = VK_NULL_HANDLE;
  nvvk::Buffer          m_ubo{};  // MVP uniform buffer
};


int main(int argc, char** argv)
{
  nvutils::ProfilerManager   profilerManager;
  nvutils::ParameterRegistry parameterRegistry;
  nvutils::ParameterParser   parameterParser;

  // setup sample element
  SampleElement::Info sampleInfo = {
      .profilerManager   = &profilerManager,
      .parameterRegistry = &parameterRegistry,
  };
  std::shared_ptr<SampleElement> sampleElement = std::make_shared<SampleElement>(sampleInfo);

  // setup logger element, `true` means shown by default
  // we add it early so outputs are captured early on, you might want to defer this to a later timer.
  std::shared_ptr<nvapp::ElementLogger> elementLogger = std::make_shared<nvapp::ElementLogger>(true);
  nvutils::Logger::getInstance().setLogCallback([&](nvutils::Logger::LogLevel logLevel, const std::string& text) {
    elementLogger->addLog(logLevel, "%s", text.c_str());
  });

  nvvk::ContextInitInfo vkSetup{
      .instanceExtensions = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME},
      .deviceExtensions   = {{VK_KHR_SWAPCHAIN_EXTENSION_NAME}},
  };

  // let's add a command-line option to enable/disable validation layers
  parameterRegistry.add({"validation"}, &vkSetup.enableValidationLayers);
  parameterRegistry.add({"verbose"}, &vkSetup.verbose);
  // as well as an option to force the vulkan device based on canonical index
  parameterRegistry.add({"forcedevice"}, &vkSetup.forceGPU);

  // add all parameters to the parser
  parameterParser.add(parameterRegistry);

  // and then parse command line
  parameterParser.parse(argc, argv);

  nvvk::addSurfaceExtensions(vkSetup.instanceExtensions);
  nvvk::Context vkContext;
  if(vkContext.init(vkSetup) != VK_SUCCESS)
  {
    LOGE("Error in Vulkan context creation\n");
    return 1;
  }

  nvapp::ApplicationCreateInfo appInfo;
  appInfo.name           = "The Empty Example";
  appInfo.useMenu        = true;
  appInfo.instance       = vkContext.getInstance();
  appInfo.device         = vkContext.getDevice();
  appInfo.physicalDevice = vkContext.getPhysicalDevice();
  appInfo.queues         = vkContext.getQueueInfos();
  appInfo.dockSetup      = [](ImGuiID viewportID) {
    // right side panel container
    ImGuiID settingID = ImGui::DockBuilderSplitNode(viewportID, ImGuiDir_Right, 0.25F, nullptr, &viewportID);
    ImGui::DockBuilderDockWindow("Settings", settingID);

    // bottom panel container
    ImGuiID loggerID = ImGui::DockBuilderSplitNode(viewportID, ImGuiDir_Down, 0.35F, nullptr, &viewportID);
    ImGui::DockBuilderDockWindow("Log", loggerID);
    ImGuiID profilerID = ImGui::DockBuilderSplitNode(loggerID, ImGuiDir_Right, 0.4F, nullptr, &loggerID);
    ImGui::DockBuilderDockWindow("Profiler", profilerID);
  };

  // Create the application
  nvapp::Application app;
  app.init(appInfo);

  // Camera element (must be added BEFORE sampleElement for per-frame matrix update)
  auto camElement = std::make_shared<nvapp::ElementCamera>(sampleElement->getCamera());
  app.addElement(camElement);

  // add the sample main element
  app.addElement(sampleElement);
  app.addElement(std::make_shared<nvapp::ElementDefaultWindowTitle>());
  // add profiler element
  app.addElement(std::make_shared<nvapp::ElementProfiler>(&profilerManager));
  // add logger element
  app.addElement(elementLogger);

  LOGI("%s", "Wohoo let's run this sample!\n");

  // enter the main loop
  app.run();

  // Cleanup in reverse order
  app.deinit();
  vkContext.deinit();

  return 0;
}
