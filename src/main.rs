use env_logger;

//#[cfg(feature = "dx11")]
//extern crate gfx_backend_dx11 as back;
//#[cfg(feature = "dx12")]
//extern crate gfx_backend_dx12 as back;
//#[cfg(feature = "metal")]
//extern crate gfx_backend_metal as back;
//#[cfg(feature = "vulkan")]
use gfx_backend_vulkan as gfx_backend;

use gfx_hal::Instance;
use gfx_hal::device::Device;
use gfx_hal::queue::family::QueueFamily;
use gfx_hal::adapter::PhysicalDevice;
use shaderc;
use std::mem::size_of;
use nalgebra_glm as glm;

const GFX_APP_NAME: &'static str = "ofl-renderer";
const GFX_BACKEND_VERSION: u32 = 1;
const QUEUE_COUNT: usize = 1;
const VIEWPORT_WIDTH: u32 = 1024;
const VIEWPORT_HEIGHT: u32 = 1024;
const VERTEX_SHADER_SOURCE: &'static str = include_str!("shader.vert.glsl");
const FRAGMENT_SHADER_SOURCE: &'static str = include_str!("shader.frag.glsl");

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct Vertex {
  pub position: [f32; 3],
  pub color: [f32; 3],
}

fn main() {
  env_logger::init();

  // For now this just panics if you didn't pass numbers. Could add proper error handling.
//  if std::env::args().len() == 1 {
//    panic!("You must pass a list of positive integers!")
//  }
//  let numbers: Vec<u32> = std::env::args()
//    .skip(1)
//    .map(|s| u32::from_str(&s).expect("You must pass a list of positive integers!"))
//    .collect();
//  let stride = std::mem::size_of::<u32>() as u64;

  let instance: gfx_backend::Instance = gfx_backend::Instance::create(GFX_APP_NAME, GFX_BACKEND_VERSION);

  let adapter: gfx_hal::Adapter<gfx_backend::Backend> = instance
    .enumerate_adapters()
    .into_iter()
    .find(|a| {
      a.queue_families
        .iter()
        .any(|qf| qf.supports_graphics())
    })
    .expect("Failed to find a compatible GPU adapter!");

  let physical_device: &gfx_backend::PhysicalDevice = &adapter.physical_device;

  let (device, mut queue_group): (gfx_backend::Device, gfx_hal::QueueGroup<gfx_backend::Backend, gfx_hal::queue::capability::Graphics>) = adapter
    .open_with::<_, gfx_hal::queue::capability::Graphics>(QUEUE_COUNT, |_qf| true)
    .expect("Failed to open GPU device");

  let mut command_pool = unsafe {
    device
      .create_command_pool_typed(&queue_group, gfx_hal::pool::CommandPoolCreateFlags::RESET_INDIVIDUAL)
      .expect("Failed to create command pool")
  };

  let cmd_queue = &mut queue_group.queues[0];


  let memory_types = physical_device
    .memory_properties()
    .memory_types;
  // Prepare vertex and index buffers
  let vertices: [Vertex; 3] = [
    Vertex { position: [1.0, 1.0, 0.0], color: [1.0, 0.0, 0.0] },
    Vertex { position: [-1.0, 1.0, 0.0], color: [0.0, 1.0, 0.0] },
    Vertex { position: [0.0, -1.0, 0.0], color: [0.0, 0.0, 1.0] },
  ];
  let indices: [u32; 3] = [0, 1, 2];

  let vertex_buffer_size = size_of::<[Vertex; 3]>();
  let index_buffer_size = size_of::<[u32; 3]>();

  let extent = gfx_hal::image::Extent { width: VIEWPORT_WIDTH, height: VIEWPORT_HEIGHT, depth: 1 };

  // WRITE THE TRIANGLE DATA
  let ((vertex_buffer, vertex_memory), (index_buffer, index_memory)) = {
    let (staging_memory, staging_buffer, staging_capacity) = unsafe {
      create_buffer::<gfx_backend::Backend>(
        &device,
        gfx_hal::buffer::Usage::TRANSFER_SRC,
        gfx_hal::memory::Properties::CPU_VISIBLE | gfx_hal::memory::Properties::COHERENT,
        vertex_buffer_size,
        &memory_types,
      )
    };

    unsafe {
      let mut staging_mapping: gfx_hal::mapping::Writer<gfx_backend::Backend, f32> = device
        .acquire_mapping_writer(&staging_memory, 0..staging_capacity)
        .expect("Failed to acquire mapping writer");
      let staging_data: [f32; 18] = [
        vertices[0].position[0], vertices[0].position[1], vertices[0].position[2], vertices[0].color[0], vertices[0].color[1], vertices[0].color[2],
        vertices[1].position[0], vertices[1].position[1], vertices[1].position[2], vertices[1].color[0], vertices[1].color[1], vertices[1].color[2],
        vertices[2].position[0], vertices[2].position[1], vertices[2].position[2], vertices[2].color[0], vertices[2].color[1], vertices[2].color[2],
      ];
      staging_mapping[..staging_data.len()].copy_from_slice(&staging_data);
      device
        .release_mapping_writer(staging_mapping)
        .expect("Failed to release mapping writer");
    }

    // (gfx_backend::native::Memory, gfx_backend::native::Buffer, u64)
    let (vertex_memory, vertex_buffer, _vertex_capacity) = unsafe {
      create_buffer::<gfx_backend::Backend>(
        &device,
        gfx_hal::buffer::Usage::VERTEX | gfx_hal::buffer::Usage::TRANSFER_DST,
        gfx_hal::memory::Properties::DEVICE_LOCAL,
        vertex_buffer_size,
        &memory_types,
      )
    };

    unsafe {
      let mut copy_cmd = command_pool.acquire_command_buffer::<gfx_hal::command::OneShot>();
      copy_cmd.begin();
      copy_cmd.copy_buffer(
        &staging_buffer,
        &vertex_buffer,
        &[gfx_hal::command::BufferCopy { src: 0, dst: 0, size: vertex_buffer_size as u64 }],
      );
      copy_cmd.finish();
      let copy_fence = device.create_fence(false).expect("Failed to create fence");
      cmd_queue.submit_nosemaphores(Some(&copy_cmd), Some(&copy_fence));
      device.wait_for_fence(&copy_fence, core::u64::MAX).expect("Failed to wait for fence");
      device.destroy_fence(copy_fence);
    }

    unsafe {
      device.destroy_buffer(staging_buffer);
      device.free_memory(staging_memory);
    }

    let (staging_memory, staging_buffer, staging_capacity) = unsafe {
      create_buffer::<gfx_backend::Backend>(
        &device,
        gfx_hal::buffer::Usage::TRANSFER_SRC,
        gfx_hal::memory::Properties::CPU_VISIBLE | gfx_hal::memory::Properties::COHERENT,
        index_buffer_size,
        &memory_types,
      )
    };

    unsafe {
      let mut staging_mapping: gfx_hal::mapping::Writer<gfx_backend::Backend, u32> = device
        .acquire_mapping_writer(&staging_memory, 0..staging_capacity)
        .expect("Failed to acquire mapping writer");
      staging_mapping[..indices.len()].copy_from_slice(&indices);
      device
        .release_mapping_writer(staging_mapping)
        .expect("Failed to release mapping writer");
    }

    let (index_memory, index_buffer, _index_capacity) = unsafe {
      create_buffer::<gfx_backend::Backend>(
        &device,
        gfx_hal::buffer::Usage::INDEX | gfx_hal::buffer::Usage::TRANSFER_DST,
        gfx_hal::memory::Properties::DEVICE_LOCAL,
        index_buffer_size,
        &memory_types,
      )
    };

    unsafe {
      let mut copy_cmd = command_pool.acquire_command_buffer::<gfx_hal::command::OneShot>();
      copy_cmd.begin();
      copy_cmd.copy_buffer(
        &staging_buffer,
        &index_buffer,
        &[gfx_hal::command::BufferCopy { src: 0, dst: 0, size: index_buffer_size as u64 }],
      );
      copy_cmd.finish();
      let copy_fence = device.create_fence(false).expect("Failed to create fence");
      cmd_queue.submit_nosemaphores(Some(&copy_cmd), Some(&copy_fence));
      device.wait_for_fence(&copy_fence, core::u64::MAX).expect("Failed to wait for fence");
      device.destroy_fence(copy_fence);
    }

    unsafe {
      device.destroy_buffer(staging_buffer);
      device.free_memory(staging_memory);
    }

    ((vertex_buffer, vertex_memory), (index_buffer, index_memory))
  };

  let color_format = gfx_hal::format::Format::Rgba8Unorm;
  let depth_format = get_supported_depth_format(physical_device).expect("Failed to find supported depth format");

  // Create attachments
  let ((color_image, color_image_memory, color_image_view), (depth_image, depth_image_memory, depth_image_view)) = unsafe {
    let color_mip_levels: gfx_hal::image::Level = 1;
    let color_samples: gfx_hal::image::NumSamples = 1;
    let color_image_kind = gfx_hal::image::Kind::D2(VIEWPORT_WIDTH, VIEWPORT_HEIGHT, 1, color_samples);
    let mut color_image = device
      .create_image(
        color_image_kind,
        color_mip_levels,
        color_format,
        gfx_hal::image::Tiling::Optimal,
        gfx_hal::image::Usage::COLOR_ATTACHMENT | gfx_hal::image::Usage::TRANSFER_SRC,
        gfx_hal::image::ViewCapabilities::empty(),
      )
      .expect("Failed to create color image");

    let color_image_requirements = device.get_image_requirements(&color_image);
    let color_image_memory_type_id = get_memory_type_id(&memory_types, gfx_hal::memory::Properties::DEVICE_LOCAL, color_image_requirements.type_mask);
    let color_image_memory = device.allocate_memory(color_image_memory_type_id, color_image_requirements.size).expect("Failed to allocate color image memory");
    device
      .bind_image_memory(&color_image_memory, 0, &mut color_image)
      .expect("Failed to bind image memory");

    let color_image_view = device
      .create_image_view(
        &color_image,
        gfx_hal::image::ViewKind::D2,
        color_format,
        gfx_hal::format::Swizzle::NO,
        gfx_hal::image::SubresourceRange {
          aspects: gfx_hal::format::Aspects::COLOR,
          layers: std::ops::Range { start: 0, end: 1 },
          levels: std::ops::Range { start: 0, end: 1 },
        },
      )
      .expect("Failed to create color image view");

    let depth_mip_levels: gfx_hal::image::Level = 1;
    let depth_samples: gfx_hal::image::NumSamples = 1;
    let depth_image_kind = gfx_hal::image::Kind::D2(VIEWPORT_WIDTH, VIEWPORT_HEIGHT, 1, depth_samples);
    let mut depth_image = device
      .create_image(
        depth_image_kind,
        depth_mip_levels,
        depth_format,
        gfx_hal::image::Tiling::Optimal,
        gfx_hal::image::Usage::DEPTH_STENCIL_ATTACHMENT,
        gfx_hal::image::ViewCapabilities::empty(),
      )
      .expect("Failed to create depth image");

    let depth_image_requirements = device.get_image_requirements(&depth_image);
    let depth_image_memory_type_id = get_memory_type_id(&memory_types, gfx_hal::memory::Properties::DEVICE_LOCAL, depth_image_requirements.type_mask);
    let depth_image_memory = device.allocate_memory(depth_image_memory_type_id, depth_image_requirements.size).expect("Failed to allocate depth image memory");
    device
      .bind_image_memory(&depth_image_memory, 0, &mut depth_image)
      .expect("Failed to bind image memory");

    let depth_image_view = device
      .create_image_view(
        &depth_image,
        gfx_hal::image::ViewKind::D2,
        depth_format,
        gfx_hal::format::Swizzle::NO,
        gfx_hal::image::SubresourceRange {
          aspects: gfx_hal::format::Aspects::DEPTH | gfx_hal::format::Aspects::STENCIL,
          layers: std::ops::Range { start: 0, end: 1 },
          levels: std::ops::Range { start: 0, end: 1 },
        },
      )
      .expect("Failed to create color image view");

    ((color_image, color_image_memory, color_image_view), (depth_image, depth_image_memory, depth_image_view))
  };

  //  Create renderpass
  let (frame_buffer, render_pass) = unsafe {
    let color_attachment: gfx_hal::pass::Attachment = gfx_hal::pass::Attachment {
      format: Some(color_format),
      samples: 1,
      ops: gfx_hal::pass::AttachmentOps {
        load: gfx_hal::pass::AttachmentLoadOp::Clear,
        store: gfx_hal::pass::AttachmentStoreOp::Store,
      },
      stencil_ops: gfx_hal::pass::AttachmentOps {
        load: gfx_hal::pass::AttachmentLoadOp::DontCare,
        store: gfx_hal::pass::AttachmentStoreOp::DontCare,
      },
      layouts: std::ops::Range { start: gfx_hal::image::Layout::Undefined, end: gfx_hal::image::Layout::TransferSrcOptimal },
    };
    let depth_attachment: gfx_hal::pass::Attachment = gfx_hal::pass::Attachment {
      format: Some(depth_format),
      samples: 1,
      ops: gfx_hal::pass::AttachmentOps {
        load: gfx_hal::pass::AttachmentLoadOp::Clear,
        store: gfx_hal::pass::AttachmentStoreOp::DontCare,
      },
      stencil_ops: gfx_hal::pass::AttachmentOps {
        load: gfx_hal::pass::AttachmentLoadOp::DontCare,
        store: gfx_hal::pass::AttachmentStoreOp::DontCare,
      },
      layouts: std::ops::Range { start: gfx_hal::image::Layout::Undefined, end: gfx_hal::image::Layout::DepthStencilAttachmentOptimal },
    };
    let attachments = [color_attachment, depth_attachment];

    let color_ref: gfx_hal::pass::AttachmentRef = (0, gfx_hal::image::Layout::ColorAttachmentOptimal);
    let depth_ref: gfx_hal::pass::AttachmentRef = (1, gfx_hal::image::Layout::DepthStencilAttachmentOptimal);

    let subpass_desc: gfx_hal::pass::SubpassDesc = gfx_hal::pass::SubpassDesc {
      colors: &[color_ref],
      depth_stencil: Some(&depth_ref),
      inputs: &[],
      resolves: &[],
      preserves: &[],
    };

    let dep0: gfx_hal::pass::SubpassDependency = gfx_hal::pass::SubpassDependency {
      passes: std::ops::Range { start: gfx_hal::pass::SubpassRef::External, end: gfx_hal::pass::SubpassRef::Pass(0) },
      stages: std::ops::Range { start: gfx_hal::pso::PipelineStage::BOTTOM_OF_PIPE, end: gfx_hal::pso::PipelineStage::COLOR_ATTACHMENT_OUTPUT },
      accesses: std::ops::Range { start: gfx_hal::image::Access::MEMORY_READ, end: gfx_hal::image::Access::COLOR_ATTACHMENT_READ | gfx_hal::image::Access::COLOR_ATTACHMENT_WRITE },
    };

    let dep1: gfx_hal::pass::SubpassDependency = gfx_hal::pass::SubpassDependency {
      passes: std::ops::Range { start: gfx_hal::pass::SubpassRef::Pass(0), end: gfx_hal::pass::SubpassRef::External },
      stages: std::ops::Range { start: gfx_hal::pso::PipelineStage::COLOR_ATTACHMENT_OUTPUT, end: gfx_hal::pso::PipelineStage::BOTTOM_OF_PIPE },
      accesses: std::ops::Range { start: gfx_hal::image::Access::COLOR_ATTACHMENT_READ | gfx_hal::image::Access::COLOR_ATTACHMENT_WRITE, end: gfx_hal::image::Access::MEMORY_READ },
    };

    let dependencies = [dep0, dep1];

    let render_pass = device
      .create_render_pass(
        &attachments,
        &[subpass_desc],
        &dependencies,
      )
      .expect("Failed to create render pass");

    let image_views = vec![&color_image_view, &depth_image_view];

    let frame_buffer = device
      .create_framebuffer(
        &render_pass,
        image_views.into_iter(),
        gfx_hal::image::Extent { width: VIEWPORT_WIDTH, height: VIEWPORT_HEIGHT, depth: 1 },
      )
      .expect("Failed to create frame buffer");

    (frame_buffer, render_pass)
  };

  let (vertex_shader_module, fragment_shader_module, descriptor_set_layout, pipeline_layout, pipeline_cache, pipeline) = unsafe {
    let descriptor_set_layout = device
      .create_descriptor_set_layout(&[], &[])
      .expect("Failed to create descriptor set layout");

    let push_constants: Vec<(gfx_hal::pso::ShaderStageFlags, core::ops::Range<u32>)> = vec![
      (gfx_hal::pso::ShaderStageFlags::VERTEX, 0..((size_of::<glm::TMat4<f32>>() / size_of::<f32>()) as u32)),
    ];

    let pipeline_layout = device
      .create_pipeline_layout(
        &[],
        push_constants,
      )
      .expect("Failed to create pipeline layout");

    let pipeline_cache = device
      .create_pipeline_cache()
      .expect("Failed to create pipeline cache");


    let mut shader_compiler: shaderc::Compiler = shaderc::Compiler::new().expect("Failed to create shader");
    let vertex_compile_artifact: shaderc::CompilationArtifact = shader_compiler
      .compile_into_spirv(
        VERTEX_SHADER_SOURCE,
        shaderc::ShaderKind::Vertex,
        "shader.vert",
        "main",
        None,
      )
      .expect("Failed to compile vertex shader");
    let fragment_compile_artifact: shaderc::CompilationArtifact = shader_compiler
      .compile_into_spirv(
        FRAGMENT_SHADER_SOURCE,
        shaderc::ShaderKind::Fragment,
        "shader.frag",
        "main",
        None,
      )
      .expect("Failed to compile fragment shader");
    let vertex_shader_module = {
      device
        .create_shader_module(vertex_compile_artifact.as_binary_u8())
        .expect("Failed to create shader module")
    };
    let fragment_shader_module = {
      device
        .create_shader_module(fragment_compile_artifact.as_binary_u8())
        .expect("Failed to create fragment module")
    };

    let shaders = gfx_hal::pso::GraphicsShaderSet {
      vertex: gfx_hal::pso::EntryPoint {
        entry: "main",
        module: &vertex_shader_module,
        specialization: gfx_hal::pso::Specialization { constants: &[], data: &[] },
      },
      hull: None,
      domain: None,
      geometry: None,
      fragment: Some(gfx_hal::pso::EntryPoint {
        entry: "main",
        module: &fragment_shader_module,
        specialization: gfx_hal::pso::Specialization { constants: &[], data: &[] },
      }),
    };

    let rasterizer = gfx_hal::pso::Rasterizer {
      depth_clamping: false,
      polygon_mode: gfx_hal::pso::PolygonMode::Fill,
      cull_face: gfx_hal::pso::Face::BACK,
      front_face: gfx_hal::pso::FrontFace::Clockwise,
      depth_bias: None,
      conservative: false,
    };

    let vertex_buffers: Vec<gfx_hal::pso::VertexBufferDesc> = vec![gfx_hal::pso::VertexBufferDesc {
      binding: 0,
      stride: (size_of::<Vertex>()) as u32,
      rate: 0,
    }];
    let attributes: Vec<gfx_hal::pso::AttributeDesc> = vec![
      // position
      gfx_hal::pso::AttributeDesc {
        binding: 0,
        location: 0,
        element: gfx_hal::pso::Element { format: gfx_hal::format::Format::Rgb32Float, offset: 0 },
      },
      // color
      gfx_hal::pso::AttributeDesc {
        binding: 0,
        location: 1,
        element: gfx_hal::pso::Element { format: gfx_hal::format::Format::Rgb32Float, offset: 4 * 3 },
      },
    ];

    let input_assembler: gfx_hal::pso::InputAssemblerDesc = gfx_hal::pso::InputAssemblerDesc::new(gfx_hal::Primitive::TriangleList);

    let blender = {
      let blend_state = gfx_hal::pso::BlendState::On {
        color: gfx_hal::pso::BlendOp::Add {
          src: gfx_hal::pso::Factor::One,
          dst: gfx_hal::pso::Factor::Zero,
        },
        alpha: gfx_hal::pso::BlendOp::Add {
          src: gfx_hal::pso::Factor::One,
          dst: gfx_hal::pso::Factor::Zero,
        },
      };
      gfx_hal::pso::BlendDesc {
        logic_op: Some(gfx_hal::pso::LogicOp::Copy),
        targets: vec![gfx_hal::pso::ColorBlendDesc(gfx_hal::pso::ColorMask::ALL, blend_state)],
      }
    };

    let depth_stencil = gfx_hal::pso::DepthStencilDesc {
      depth: gfx_hal::pso::DepthTest::On { fun: gfx_hal::pso::Comparison::LessEqual, write: true },
      depth_bounds: false,
      stencil: gfx_hal::pso::StencilTest::Off,
    };

    let multisampling: Option<gfx_hal::pso::Multisampling> = None;

    let baked_states = gfx_hal::pso::BakedStates {
      viewport: Some(gfx_hal::pso::Viewport {
        rect: extent.rect(),
        depth: (0.0..1.0),
      }),
      scissor: Some(extent.rect()),
      blend_color: None,
      depth_bounds: None,
    };

    let pipeline_flags: gfx_hal::pso::PipelineCreationFlags = gfx_hal::pso::PipelineCreationFlags::empty();

    let pipeline_desc = gfx_hal::pso::GraphicsPipelineDesc {
      shaders,
      rasterizer,
      vertex_buffers,
      attributes,
      input_assembler,
      blender,
      depth_stencil,
      multisampling,
      baked_states,
      layout: &pipeline_layout,
      subpass: gfx_hal::pass::Subpass {
        index: 0,
        main_pass: &render_pass,
      },
      flags: pipeline_flags,
      parent: gfx_hal::pso::BasePipeline::None,
    };

    let pipeline = device
      .create_graphics_pipeline(&pipeline_desc, Some(&pipeline_cache))
      .expect("Failed to create pipeline");

    (vertex_shader_module, fragment_shader_module, descriptor_set_layout, pipeline_layout, pipeline_cache, pipeline)
  };

  unsafe {
    let mut command_buffer = command_pool.acquire_command_buffer::<gfx_hal::command::OneShot>();

    command_buffer.begin();

    {
      let clear_values = [
        gfx_hal::command::ClearValue::Color(gfx_hal::command::ClearColor::Float([0.0, 0.0, 0.2, 1.0])),
        gfx_hal::command::ClearValue::DepthStencil(gfx_hal::command::ClearDepthStencil(1.0, 0)),
      ];
      let mut encoder: gfx_hal::command::RenderPassInlineEncoder<_> = command_buffer.begin_render_pass_inline(
        &render_pass,
        &frame_buffer,
        extent.rect(),
        clear_values.iter(),
      );

      let viewports = vec![gfx_hal::pso::Viewport { rect: extent.rect(), depth: (0.0..1.0) }];
      encoder.set_viewports(0, viewports);

      let scissors = vec![extent.rect()];
      encoder.set_scissors(0, scissors);

      encoder.bind_graphics_pipeline(&pipeline);

      encoder.bind_vertex_buffers(0, vec![(&vertex_buffer, 0)]);
      encoder.bind_index_buffer(gfx_hal::buffer::IndexBufferView {
        buffer: &index_buffer,
        offset: 0,
        index_type: gfx_hal::IndexType::U32,
      });

      let pos = vec![
        glm::vec3(-1.5f32, 0.0f32, -4.0f32),
        glm::vec3(0.0f32, 0.0f32, -2.5f32),
        glm::vec3(1.5f32, 0.0f32, -4.0f32),
      ];

      for v in pos {
        let perspective = glm::perspective(
          1.0, // glm::radians(60.0f32),
          (VIEWPORT_WIDTH as f32) / (VIEWPORT_HEIGHT as f32),
          0.1f32,
          256.0f32,
        );

        let identiy4: glm::TMat4<f32> = glm::identity();

        let mvp_matrix: glm::TMat4<f32> = perspective * glm::translate(&identiy4, &v);

        let mvp_matrix_bits: Vec<u32> = mvp_matrix.data.iter().map(|x| x.to_bits()).collect();

        encoder.push_graphics_constants(
          &pipeline_layout,
          gfx_hal::pso::ShaderStageFlags::VERTEX,
          0,
          &mvp_matrix_bits[..],
        );
        encoder.draw_indexed(0..3, 0, 0..1);
      }
    }

    command_buffer.finish();

    let cmd_fence = device.create_fence(false).expect("Failed to create fence");
    cmd_queue.submit_nosemaphores(Some(&command_buffer), Some(&cmd_fence));
    device.wait_for_fence(&cmd_fence, core::u64::MAX).expect("Failed to wait for fence");
    device.destroy_fence(cmd_fence);

    device
      .wait_idle()
      .expect("Failed to wait for device to be idle");
  }

  let (dst_image, dst_image_memory, dst_image_data) = unsafe {
    let mut dst_image = device
      .create_image(
        gfx_hal::image::Kind::D2(VIEWPORT_WIDTH, VIEWPORT_HEIGHT, 1, 1),
        1,
        color_format,
        gfx_hal::image::Tiling::Linear,
        gfx_hal::image::Usage::TRANSFER_DST,
        gfx_hal::image::ViewCapabilities::empty(),
      )
      .expect("Failed to create color image");

    let dst_image_requirements = device.get_image_requirements(&dst_image);
    let dst_image_memory_type_id = get_memory_type_id(&memory_types, gfx_hal::memory::Properties::CPU_VISIBLE | gfx_hal::memory::Properties::COHERENT, dst_image_requirements.type_mask);
    let dst_image_memory = device.allocate_memory(dst_image_memory_type_id, dst_image_requirements.size).expect("Failed to allocate dst image memory");
    device
      .bind_image_memory(&dst_image_memory, 0, &mut dst_image)
      .expect("Failed to bind image memory");

    {
      let mut copy_cmd = command_pool.acquire_command_buffer::<gfx_hal::command::OneShot>();
      copy_cmd.begin();

      {
        let src_state: gfx_hal::image::State = (gfx_hal::image::Access::empty(), gfx_hal::image::Layout::Undefined);
        let dst_state: gfx_hal::image::State = (gfx_hal::image::Access::TRANSFER_WRITE, gfx_hal::image::Layout::TransferDstOptimal);
        let barrier: gfx_hal::memory::Barrier<gfx_backend::Backend> = gfx_hal::memory::Barrier::Image {
          states: (src_state..dst_state),
          target: &dst_image,
          families: None,
          range: gfx_hal::image::SubresourceRange {
            aspects: gfx_hal::format::Aspects::COLOR,
            layers: 0..1,
            levels: 0..1,
          },
        };
        copy_cmd.pipeline_barrier(
          gfx_hal::pso::PipelineStage::TRANSFER..gfx_hal::pso::PipelineStage::TRANSFER,
          gfx_hal::memory::Dependencies::empty(),
          Some(barrier),
        );
      }

      let image_copy_regions: gfx_hal::command::ImageCopy = gfx_hal::command::ImageCopy {
        src_subresource: gfx_hal::image::SubresourceLayers {
          aspects: gfx_hal::format::Aspects::COLOR,
          level: 0,
          layers: 0..1,
        },
        src_offset: gfx_hal::image::Offset { x: 0, y: 0, z: 0 },
        dst_subresource: gfx_hal::image::SubresourceLayers {
          aspects: gfx_hal::format::Aspects::COLOR,
          level: 0,
          layers: 0..1,
        },
        dst_offset: gfx_hal::image::Offset { x: 0, y: 0, z: 0 },
        extent: extent.clone(),
      };
      copy_cmd.copy_image(
        &color_image,
        gfx_hal::image::Layout::TransferSrcOptimal,
        &dst_image,
        gfx_hal::image::Layout::TransferDstOptimal,
        Some(&image_copy_regions),
      );

      {
        let src_state: gfx_hal::image::State = (gfx_hal::image::Access::TRANSFER_WRITE, gfx_hal::image::Layout::TransferDstOptimal);
        let dst_state: gfx_hal::image::State = (gfx_hal::image::Access::MEMORY_READ, gfx_hal::image::Layout::General);
        let barrier: gfx_hal::memory::Barrier<gfx_backend::Backend> = gfx_hal::memory::Barrier::Image {
          states: (src_state..dst_state),
          target: &dst_image,
          families: None,
          range: gfx_hal::image::SubresourceRange {
            aspects: gfx_hal::format::Aspects::COLOR,
            layers: 0..1,
            levels: 0..1,
          },
        };
        copy_cmd.pipeline_barrier(
          gfx_hal::pso::PipelineStage::TRANSFER..gfx_hal::pso::PipelineStage::TRANSFER,
          gfx_hal::memory::Dependencies::empty(),
          Some(barrier),
        );
      }

      copy_cmd.finish();

      let copy_fence = device.create_fence(false).expect("Failed to create fence");
      cmd_queue.submit_nosemaphores(Some(&copy_cmd), Some(&copy_fence));
      device.wait_for_fence(&copy_fence, core::u64::MAX).expect("Failed to wait for fence");
      device.destroy_fence(copy_fence);
    }

    let dst_image_footprint = device.get_image_subresource_footprint(
      &dst_image,
      gfx_hal::image::Subresource {
        aspects: gfx_hal::format::Aspects::COLOR,
        level: 0,
        layer: 0,
      },
    );

    let dst_image_data = {
      let dst_mapping: gfx_hal::mapping::Reader<gfx_backend::Backend, u8> = device
        .acquire_mapping_reader(&dst_image_memory, dst_image_footprint.slice)
        .expect("Failed to acquire mapping reader");

      let mut dst_image_data: Vec<u8> = Vec::new();

      for y in 0..(VIEWPORT_HEIGHT as usize) {
        let row_idx: usize = y * dst_image_footprint.row_pitch as usize;
        for x in 0..(VIEWPORT_WIDTH as usize) {
          let idx: usize = row_idx + 4 * x;
          dst_image_data.push(dst_mapping[idx + 0]);
          dst_image_data.push(dst_mapping[idx + 1]);
          dst_image_data.push(dst_mapping[idx + 2]);
          dst_image_data.push(dst_mapping[idx + 3]);
        }
      }

      device
        .release_mapping_reader(dst_mapping);

      dst_image_data
    };

    (dst_image, dst_image_memory, dst_image_data)
  };

  {
    let pam_file = ::std::fs::File::create("out.pam").expect("Failed to create actual AST file");
    let mut pam_writer = ::std::io::BufWriter::new(pam_file);
    write_pam(&mut pam_writer, &dst_image_data).expect("Failed to write PAM");
  }

  unsafe {
//    device.destroy_image_view(dst_image_view);
    device.destroy_image(dst_image);
    device.free_memory(dst_image_memory);

    device.destroy_graphics_pipeline(pipeline);
    device.destroy_pipeline_cache(pipeline_cache);
    device.destroy_pipeline_layout(pipeline_layout);
    device.destroy_descriptor_set_layout(descriptor_set_layout);
    device.destroy_shader_module(fragment_shader_module);
    device.destroy_shader_module(vertex_shader_module);

    device.destroy_framebuffer(frame_buffer);
    device.destroy_render_pass(render_pass);

    device.destroy_image_view(color_image_view);
    device.destroy_image(color_image);
    device.free_memory(color_image_memory);
    device.destroy_image_view(depth_image_view);
    device.destroy_image(depth_image);
    device.free_memory(depth_image_memory);

    device.destroy_buffer(index_buffer);
    device.free_memory(index_memory);
    device.destroy_buffer(vertex_buffer);
    device.free_memory(vertex_memory);

    device.destroy_command_pool(command_pool.into_raw());
  }

  dbg!("done");
}

fn get_memory_type_id(memory_types: &[gfx_hal::MemoryType], memory_properties: gfx_hal::memory::Properties, mem_type_mask: u64) -> gfx_hal::MemoryTypeId {
  memory_types
    .into_iter()
    .enumerate()
    .position(|(id, memory_type)| {
// Typemask is a bitset where the bit `2^id` indicates compatibility with the memory type with
// the corresponding `id`.
      (mem_type_mask & (1 << id) != 0) & &memory_type.properties.contains(memory_properties)
    })
    .expect("Failed to find compatible memory type")
    .into()
}

unsafe fn create_buffer<B: gfx_hal::Backend>(
  device: &B::Device,
  usage: gfx_hal::buffer::Usage,
  memory_properties: gfx_hal::memory::Properties,
  size: usize,
  memory_types: &[gfx_hal::MemoryType],
//  properties: memory::Properties,
//  usage: buffer::Usage,
//  stride: u64,
//  len: u64,
) -> (B::Memory, B::Buffer, u64) {

// // Create the buffer handle
// VkBufferCreateInfo bufferCreateInfo = vks::initializers::bufferCreateInfo(usageFlags, size);
// bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
// VK_CHECK_RESULT(vkCreateBuffer(device, &bufferCreateInfo, nullptr, buffer));
  let mut buffer = device.create_buffer(size as u64, usage).expect("Failed to create buffer");

  let requirements: gfx_hal::memory::Requirements = device.get_buffer_requirements(&buffer);

  let mem_type: gfx_hal::MemoryTypeId = get_memory_type_id(memory_types, memory_properties, requirements.type_mask);

  let memory = device.allocate_memory(mem_type, requirements.size).expect("Failed to allocate memory");
  device.bind_buffer_memory(&memory, 0, &mut buffer).expect("Failed to bind buffer to memory");

  (memory, buffer, requirements.size)
}

fn get_supported_depth_format(physical_device: &gfx_backend::PhysicalDevice) -> Option<gfx_hal::format::Format> {
  let depth_formats = [
    gfx_hal::format::Format::D32FloatS8Uint,
    gfx_hal::format::Format::D32Float,
    gfx_hal::format::Format::D24UnormS8Uint,
    gfx_hal::format::Format::D16UnormS8Uint,
    gfx_hal::format::Format::D16Unorm,
  ];

  for format in depth_formats.into_iter() {
    let format_properties = physical_device.format_properties(Some(*format));
    if format_properties.optimal_tiling.contains(gfx_hal::format::ImageFeature::DEPTH_STENCIL_ATTACHMENT) {
      return Some(*format);
    }
  }

  Option::None
}

pub fn write_pam<W>(writer: &mut W, image_data: &[u8]) -> ::std::io::Result<usize> where W: ::std::io::Write {
  writer.write(b"P7\n")?;
  writer.write(b"WIDTH 1024\n")?;
  writer.write(b"HEIGHT 1024\n")?;
  writer.write(b"DEPTH 4\n")?;
  writer.write(b"MAXVAL 255\n")?;
  writer.write(b"TUPLTYPE RGB_ALPHA\n")?;
  writer.write(b"ENDHDR\n")?;
  writer.write(&image_data)
}
