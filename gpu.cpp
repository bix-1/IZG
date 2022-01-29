// gpu.cpp
// Project for course IZG @ FIT BUT
// Project Author:
//      Tomáš Milet, imilet@fit.vutbr.cz
// Implementation of Assignment:
//      Jakub Bartko, xbartk07
//      xbartk07@stud.fit.vutbr.cz
// FIT VUT Brno, 23.05.2020

// Description:
//    Implementation of primitive GPU
// Usage:
//    ./izgProject  [ in ../build/ ]

/*  === === === === ===[ Assignment ]=== === === === ===
    Implement primitive GPU.
      (Using said GPU) visualize the rabbit model
      using Phong reflection model and Phong shading.
    For complete assignment see ../doc/html/index.html
    === === === === === === ==== === === === === === === */

/*!
 * @file
 * @brief This file contains implementation of gpu
 *
 * @author Tomáš Milet, imilet@fit.vutbr.cz
 */

#include <student/gpu.hpp>

/// \addtogroup gpu_init
/// @{

/**
 * @brief Constructor of GPU
 */
GPU::GPU() {}

/**
 * @brief Destructor of GPU
 */
GPU::~GPU() {}

/// @}

/** \addtogroup buffer_tasks 01. Implementace obslužných funkcí pro buffery
 * @{
 */

/**
 * @brief This function allocates buffer on GPU.
 *
 * @param size size in bytes of new buffer on GPU.
 *
 * @return unique identificator of the buffer
 */
BufferID GPU::createBuffer(uint64_t size)
{
    auto buffer = std::make_unique<buffer_t>(size);
    auto id = reinterpret_cast<BufferID>(buffer.get());
    buffers.insert(std::make_pair(id, std::move(buffer)));

    return id;
}

/**
 * @brief This function frees allocated buffer on GPU.
 *
 * @param buffer buffer identificator
 */
void GPU::deleteBuffer(BufferID buffer)
{
    if (buffers.find(buffer) != buffers.end())
        buffers.erase(buffer);
}

/**
 * @brief This function uploads data to selected buffer on the GPU
 *
 * @param buffer buffer identificator
 * @param offset specifies the offset into the buffer's data
 * @param size specifies the size of buffer that will be uploaded
 * @param data specifies a pointer to new data
 */
void GPU::setBufferData(BufferID buffer,
                        uint64_t offset,
                        uint64_t size,
                        void const* data)
{
    if (!isBuffer(buffer) || !data)
        return;
    std::copy(
      (uint8_t*)data, (uint8_t*)data + size, buffers[buffer]->begin() + offset);
}

/**
 * @brief This function downloads data from GPU.
 *
 * @param buffer specfies buffer
 * @param offset specifies the offset into the buffer from which data will be
 * returned, measured in bytes.
 * @param size specifies data size that will be copied
 * @param data specifies a pointer to the location where buffer data is
 * returned.
 */
void GPU::getBufferData(BufferID buffer,
                        uint64_t offset,
                        uint64_t size,
                        void* data)
{
    if (!isBuffer(buffer) || !data)
        return;
    auto begin = buffers[buffer]->begin() + offset;
    std::copy(begin, begin + size, (uint8_t*)data);
}

/**
 * @brief This function tests if buffer exists
 *
 * @param buffer selected buffer id
 *
 * @return true if buffer points to existing buffer on the GPU.
 */
bool GPU::isBuffer(BufferID buffer)
{
    return buffers.find(buffer) != buffers.end();
}

/// @}

/**
 * \addtogroup vertexpuller_tasks 02. Implementace obslužných funkcí pro vertex
 * puller
 * @{
 */

/**
 * @brief This function creates new vertex puller settings on the GPU,
 *
 * @return unique vertex puller identificator
 */
ObjectID GPU::createVertexPuller()
{
    auto vp = std::make_unique<VertexPuller>();
    for (auto i = 0; i < maxAttributes; i++)
    {
        auto head = std::make_unique<Head>();
        vp->heads.push_back(std::move(head));
    }
    ObjectID id = reinterpret_cast<ObjectID>(vp.get());
    vertexPullers.insert(std::make_pair(id, std::move(vp)));

    return id;
}

/**
 * @brief This function deletes vertex puller settings
 *
 * @param vao vertex puller identificator
 */
void GPU::deleteVertexPuller(VertexPullerID vao)
{
    if (isVertexPuller(vao))
        vertexPullers.erase(vao);
}

/**
 * @brief This function sets one vertex puller reading head.
 *
 * @param vao identificator of vertex puller
 * @param head id of vertex puller head
 * @param type type of attribute
 * @param stride stride in bytes
 * @param offset offset in bytes
 * @param buffer id of buffer
 */
void GPU::setVertexPullerHead(VertexPullerID vao,
                              uint32_t head,
                              AttributeType type,
                              uint64_t stride,
                              uint64_t offset,
                              BufferID buffer)
{
    if (!isVertexPuller(vao) || head >= vertexPullers[vao]->heads.size())
        return;
    *(vertexPullers[vao]->heads[head]) = {
      .type = type, .stride = stride, .offset = offset, .buffer = buffer};
}

/**
 * @brief This function sets vertex puller indexing.
 *
 * @param vao vertex puller id
 * @param type type of index
 * @param buffer buffer with indices
 */
void GPU::setVertexPullerIndexing(VertexPullerID vao,
                                  IndexType type,
                                  BufferID buffer)
{
    if (isVertexPuller(vao))
        vertexPullers[vao]->indexing = {
          .enabled = true, .buffer = buffer, .type = type};
}

/**
 * @brief This function enables vertex puller's head.
 *
 * @param vao vertex puller
 * @param head head id
 */
void GPU::enableVertexPullerHead(VertexPullerID vao, uint32_t head)
{
    if (isVertexPuller(vao) && head < vertexPullers[vao]->heads.size())
        vertexPullers[vao]->heads[head]->enabled = true;
}

/**
 * @brief This function disables vertex puller's head
 *
 * @param vao vertex puller id
 * @param head head id
 */
void GPU::disableVertexPullerHead(VertexPullerID vao, uint32_t head)
{
    if (isVertexPuller(vao) && head < vertexPullers[vao]->heads.size())
        vertexPullers[vao]->heads[head]->enabled = false;
}

/**
 * @brief This function selects active vertex puller.
 *
 * @param vao id of vertex puller
 */
void GPU::bindVertexPuller(VertexPullerID vao)
{
    if (isVertexPuller(vao))
        activeVP = vao;
}

/**
 * @brief This function deactivates vertex puller.
 */
void GPU::unbindVertexPuller()
{
    activeVP = emptyID;
}

/**
 * @brief This function tests if vertex puller exists.
 *
 * @param vao vertex puller
 *
 * @return true, if vertex puller "vao" exists
 */
bool GPU::isVertexPuller(VertexPullerID vao)
{
    return vertexPullers.find(vao) != vertexPullers.end();
}

/// @}

/** \addtogroup program_tasks 03. Implementace obslužných funkcí pro shader
 * programy
 * @{
 */

/**
 * @brief This function creates new shader program.
 *
 * @return shader program id
 */
ProgramID GPU::createProgram()
{
    auto prg = std::make_unique<Program>();
    ProgramID id = reinterpret_cast<ProgramID>(prg.get());
    programs.insert(std::make_pair(id, std::move(prg)));

    return id;
}

/**
 * @brief This function deletes shader program
 *
 * @param prg shader program id
 */
void GPU::deleteProgram(ProgramID prg)
{
    if (isProgram(prg))
        programs.erase(prg);
}

/**
 * @brief This function attaches vertex and frament shader to shader program.
 *
 * @param prg shader program
 * @param vs vertex shader
 * @param fs fragment shader
 */
void GPU::attachShaders(ProgramID prg, VertexShader vs, FragmentShader fs)
{
    if (isProgram(prg))
        *programs[prg] = {.VS = vs, .FS = fs};
}

/**
 * @brief This function selects which vertex attributes should be interpolated
 * during rasterization into fragment attributes.
 *
 * @param prg shader program
 * @param attrib id of attribute
 * @param type type of attribute
 */
void GPU::setVS2FSType(ProgramID prg, uint32_t attrib, AttributeType type)
{
    if (isProgram(prg))
        programs[prg]->VStoFS.push_back(std::make_pair(attrib, type));
}

/**
 * @brief This function actives selected shader program
 *
 * @param prg shader program id
 */
void GPU::useProgram(ProgramID prg)
{
    if (isProgram(prg))
        activePrg = prg;
}

/**
 * @brief This function tests if selected shader program exists.
 *
 * @param prg shader program
 *
 * @return true, if shader program "prg" exists.
 */
bool GPU::isProgram(ProgramID prg)
{
    return programs.find(prg) != programs.end();
}

/**
 * @brief This function sets uniform value (1 float).
 *
 * @param prg shader program
 * @param uniformId id of uniform value (number of uniform values is stored in
 * maxUniforms variable)
 * @param d value of uniform variable
 */
void GPU::programUniform1f(ProgramID prg, uint32_t uniformId, float const& d)
{
    if (isProgram(prg) && uniformId < maxUniforms)
        programs[prg]->uniforms.uniform[uniformId].v1 = d;
}

/**
 * @brief This function sets uniform value (2 float).
 *
 * @param prg shader program
 * @param uniformId id of uniform value (number of uniform values is stored in
 * maxUniforms variable)
 * @param d value of uniform variable
 */
void GPU::programUniform2f(ProgramID prg,
                           uint32_t uniformId,
                           glm::vec2 const& d)
{
    if (isProgram(prg) && uniformId < maxUniforms)
        programs[prg]->uniforms.uniform[uniformId].v2 = d;
}

/**
 * @brief This function sets uniform value (3 float).
 *
 * @param prg shader program
 * @param uniformId id of uniform value (number of uniform values is stored in
 * maxUniforms variable)
 * @param d value of uniform variable
 */
void GPU::programUniform3f(ProgramID prg,
                           uint32_t uniformId,
                           glm::vec3 const& d)
{
    if (isProgram(prg) && uniformId < maxUniforms)
        programs[prg]->uniforms.uniform[uniformId].v3 = d;
}

/**
 * @brief This function sets uniform value (4 float).
 *
 * @param prg shader program
 * @param uniformId id of uniform value (number of uniform values is stored in
 * maxUniforms variable)
 * @param d value of uniform variable
 */
void GPU::programUniform4f(ProgramID prg,
                           uint32_t uniformId,
                           glm::vec4 const& d)
{
    if (isProgram(prg) && uniformId < maxUniforms)
        programs[prg]->uniforms.uniform[uniformId].v4 = d;
}

/**
 * @brief This function sets uniform value (4 float).
 *
 * @param prg shader program
 * @param uniformId id of uniform value (number of uniform values is stored in
 * maxUniforms variable)
 * @param d value of uniform variable
 */
void GPU::programUniformMatrix4f(ProgramID prg,
                                 uint32_t uniformId,
                                 glm::mat4 const& d)
{
    if (isProgram(prg) && uniformId < maxUniforms)
        programs[prg]->uniforms.uniform[uniformId].m4 = d;
}

/// @}

/** \addtogroup framebuffer_tasks 04. Implementace obslužných funkcí pro
 * framebuffer
 * @{
 */

/**
 * @brief This function creates framebuffer on GPU.
 *
 * @param width width of framebuffer
 * @param height height of framebuffer
 */
void GPU::createFramebuffer(uint32_t width, uint32_t height)
{
    FrameBuffer = {.width = width, .height = height};
    FrameBuffer.ColorBuffer.resize(width * height * 4); // RGBA -> 4
    FrameBuffer.DepthBuffer.resize(width * height);
}

/**
 * @brief This function deletes framebuffer.
 */
void GPU::deleteFramebuffer()
{
    FrameBuffer = {.width = 0, .height = 0};
    FrameBuffer.ColorBuffer.clear();
    FrameBuffer.DepthBuffer.clear();
}

/**
 * @brief This function resizes framebuffer.
 *
 * @param width new width of framebuffer
 * @param height new heght of framebuffer
 */
void GPU::resizeFramebuffer(uint32_t width, uint32_t height)
{
    GPU::createFramebuffer(width, height);
}

/**
 * @brief This function returns pointer to color buffer.
 *
 * @return pointer to color buffer
 */
uint8_t* GPU::getFramebufferColor()
{
    return FrameBuffer.ColorBuffer.data();
}

/**
 * @brief This function returns pointer to depth buffer.
 *
 * @return pointer to dept buffer.
 */
float* GPU::getFramebufferDepth()
{
    return FrameBuffer.DepthBuffer.data();
}

/**
 * @brief This function returns width of framebuffer
 *
 * @return width of framebuffer
 */
uint32_t GPU::getFramebufferWidth()
{
    return FrameBuffer.width;
}

/**
 * @brief This function returns height of framebuffer.
 *
 * @return height of framebuffer
 */
uint32_t GPU::getFramebufferHeight()
{
    return FrameBuffer.height;
}

/// @}

/** \addtogroup draw_tasks 05. Implementace vykreslovacích funkcí
 * Bližší informace jsou uvedeny na hlavní stránce dokumentace.
 * @{
 */

/**
 * @brief This function clears framebuffer.
 *
 * @param r red channel
 * @param g green channel
 * @param b blue channel
 * @param a alpha channel
 */
void GPU::clear(float r, float g, float b, float a)
{
    const int r_ = round(r * 255);
    const int g_ = round(g * 255);
    const int b_ = round(b * 255);
    const int a_ = round(a * 255);

    // fill color buffer
    const auto size = getFramebufferWidth() * getFramebufferHeight() * 4;
    for (auto i = 0; i < size; i += 4)
    {
        FrameBuffer.ColorBuffer[i] = r_;
        FrameBuffer.ColorBuffer[i + 1] = g_;
        FrameBuffer.ColorBuffer[i + 2] = b_;
        FrameBuffer.ColorBuffer[i + 3] = a_;
    }
    // fill depth buffer
    std::fill(FrameBuffer.DepthBuffer.begin(),
              FrameBuffer.DepthBuffer.end(),
              std::numeric_limits<float>::infinity());
}

void GPU::drawTriangles(uint32_t nofVertices)
{
    // assemble triangles -->triangleBuff
    GPU::primAssembly(nofVertices);
    // clip triangles for near-view plane
    GPU::clipping();
    // perspective division
    GPU::persDivision();
    // viewport transformation
    GPU::viewport_transf();
    // rasterization of triangles -->inFragBuff
    GPU::rasterize();
    // fragments: inFragBuff --> FrameBuffer
    GPU::perFrag_Op();

    inFragBuff.clear();
    triangleBuff.clear();
}

void GPU::perFrag_Op()
{
    for (const auto& in : inFragBuff)
    {
        auto y = floor(in.gl_FragCoord[0]);
        auto x = floor(in.gl_FragCoord[1]);
        uint32_t pos = x * GPU::getFramebufferWidth() + y;

        OutFragment out{GPU::frag_proc(in)};

        if ( // old frag depth > new frag depth
          FrameBuffer.DepthBuffer[pos] > in.gl_FragCoord[2])
        { // replace: old frag --> new frag
            for (auto i = 0; i < 4; i++)
            {
                // cut color to <0, 1>
                out.gl_FragColor[i] = std::max(0.f, out.gl_FragColor[i]);
                out.gl_FragColor[i] = std::min(1.f, out.gl_FragColor[i]);

                FrameBuffer.ColorBuffer[(pos * 4) + i] =
                  out.gl_FragColor[i] * 255;
            }
            FrameBuffer.DepthBuffer[pos] = in.gl_FragCoord[2];
        }
    }
}

OutFragment GPU::frag_proc(const InFragment& in)
{
    if (!isProgram(activePrg))
        exit(1);
    auto prg = programs[activePrg].get();
    auto fragShader = prg->FS;
    auto unif = prg->uniforms;

    OutFragment out;

    (*fragShader)(out, in, unif);

    return out;
}

void GPU::rasterize()
{
    for (auto& tr : triangleBuff)
    {
        // vertices of triangle
        auto& V0 = tr[0];
        auto& V1 = tr[1];
        auto& V2 = tr[2];
        // rectangle around triangle -- bounds [minX, maxX, minY, maxY]
        auto bounds = GPU::get_bounds(V0, V1, V2);

        // Y axis
        for (auto y = bounds[2]; y <= bounds[3]; y++)
        {
            // border points for each line
            // 0.5f accounts for middle of pixel
            glm::vec4 pL{bounds[0] + .5f, y + .5f, 0, 0}; // leftmost point
            glm::vec4 pR{bounds[1] + .5f, y + .5f, 0, 0}; // rightmost point
            // conversion for pL
            GPU::Barycentric(pL,
                             V0.gl_Position,
                             V1.gl_Position,
                             V2.gl_Position,
                             pL[0],
                             pL[1],
                             pL[2]);
            // conversion for pR
            GPU::Barycentric(pR,
                             V0.gl_Position,
                             V1.gl_Position,
                             V2.gl_Position,
                             pR[0],
                             pR[1],
                             pR[2]);

            // single step right (vec(pL, pR) divided by width of line)
            glm::vec4 mvR = pR - pL;
            mvR /= bounds[1] - bounds[0];
            // current point in barycentric coords
            glm::vec4 p = pL; // iterates from pL to pR

            // X axis
            for (auto x = bounds[0]; x <= bounds[1]; x++)
            {
                // if inside triangle
                if (0 <= p[0] && p[0] <= 1 && 0 <= p[1] && p[1] <= 1 &&
                    0 <= p[2] && p[2] <= 1)
                {
                    InFragment frag{};

                    // Fragment Attributes (interpolation)
                    auto attList = &(programs[activePrg]->VStoFS);
                    for (auto it = attList->begin(); it != attList->end(); it++)
                    {
                        auto i = it->first;
                        auto size = (uint32_t)(it->second);
                        GPU::interpol_attribs(
                          frag.attributes[i], p, i, size, V0, V1, V2);
                    }

                    // Fragment Coordinates
                    frag.gl_FragCoord[0] = x + 0.5f;
                    frag.gl_FragCoord[1] = y + 0.5f;
                    // interpolation of depth
                    auto denom = (p[0] / V0.gl_Position[3]) +
                                 (p[1] / V1.gl_Position[3]) +
                                 (p[2] / V2.gl_Position[3]);
                    frag.gl_FragCoord[2] =
                      (p[0] * V0.gl_Position[2] / V0.gl_Position[3] +
                       p[1] * V1.gl_Position[2] / V1.gl_Position[3] +
                       p[2] * V2.gl_Position[2] / V2.gl_Position[3]) /
                      denom;
                    inFragBuff.push_back(frag);
                }         // fragment creation
                p += mvR; // single step right == x++
            }             // x itr
        }                 // y itr
    }                     // triangle itr
} // GPU::rasterize()

void GPU::interpol_attribs(Attribute& att,
                           const glm::vec4& p,
                           const uint32_t pos,
                           const uint32_t size,
                           const OutVertex& A,
                           const OutVertex& B,
                           const OutVertex& C)
{
    auto h0 = A.gl_Position[3];
    auto h1 = B.gl_Position[3];
    auto h2 = C.gl_Position[3];
    auto denom = (p[0] / h0) + (p[1] / h1) + (p[2] / h2);

    switch (size)
    {
        case 1:
            att.v1 = p[0] * A.attributes[pos].v1 / h0;
            att.v1 += p[1] * B.attributes[pos].v1 / h1;
            att.v1 += p[2] * C.attributes[pos].v1 / h2;
            att.v1 /= denom;
            break;
        case 2:
            att.v2 = p[0] * A.attributes[pos].v2 / h0;
            att.v2 += p[1] * B.attributes[pos].v2 / h1;
            att.v2 += p[2] * C.attributes[pos].v2 / h2;
            att.v2 /= denom;
            break;
        case 3:
            att.v3 = p[0] * A.attributes[pos].v3 / h0;
            att.v3 += p[1] * B.attributes[pos].v3 / h1;
            att.v3 += p[2] * C.attributes[pos].v3 / h2;
            att.v3 /= denom;
            break;
        case 4:
            att.v4 = p[0] * A.attributes[pos].v4 / h0;
            att.v4 += p[1] * B.attributes[pos].v4 / h1;
            att.v4 += p[2] * C.attributes[pos].v4 / h2;
            att.v4 /= denom;
            break;
    }
}

// Transcribed from Christer Ericson's Real-Time Collision Detection
// http://realtimecollisiondetection.net/
void GPU::Barycentric(const glm::vec4 p,
                      const glm::vec4 a,
                      const glm::vec4 b,
                      const glm::vec4 c,
                      float& u,
                      float& v,
                      float& w)
{
    auto v0 = b - a;
    auto v1 = c - a;
    auto v2 = p - a;
    float d00 = glm::dot(v0, v0);
    float d01 = glm::dot(v0, v1);
    float d11 = glm::dot(v1, v1);
    float d20 = glm::dot(v2, v0);
    float d21 = glm::dot(v2, v1);
    float denom = d00 * d11 - d01 * d01;
    v = (d11 * d20 - d01 * d21) / denom;
    w = (d00 * d21 - d01 * d20) / denom;
    u = 1.0f - v - w;
}

std::vector<int> GPU::get_bounds(const OutVertex& a,
                                 const OutVertex& b,
                                 const OutVertex& c)
{
    const auto A = a.gl_Position;
    const auto B = b.gl_Position;
    const auto C = c.gl_Position;
    const auto w = GPU::getFramebufferWidth();
    const auto h = GPU::getFramebufferHeight();

    std::vector<int> out{int(std::min(A[0], std::min(B[0], C[0]))),
                         int(std::max(A[0], std::max(B[0], C[0]))),
                         int(std::min(A[1], std::min(B[1], C[1]))),
                         int(std::max(A[1], std::max(B[1], C[1])))};

    // lower limit -- 0
    for (auto& x : out)
        x = std::max(0, x);
    // upper limit -- h or w
    for (auto i = 0; i < out.size(); i++)
    {
        const auto lim = i < 2 ? w : h;
        out[i] = out[i] < lim ? out[i] : lim - 1;
    }

    return out;
}

void GPU::persDivision()
{
    for (auto& triangle : triangleBuff)
        for (auto& vertex : triangle)
            for (auto k = 0; k < 3; k++)
                vertex.gl_Position[k] /= vertex.gl_Position[3];
}

void GPU::viewport_transf()
{
    auto w = GPU::getFramebufferWidth();
    auto h = GPU::getFramebufferHeight();

    for (auto& triangle : triangleBuff)
    {
        for (auto& vx : triangle)
        {
            vx.gl_Position[0] = (vx.gl_Position[0] + 1) * w / 2;
            vx.gl_Position[1] = (vx.gl_Position[1] + 1) * h / 2;
        }
    }
}

void GPU::clipping()
{
    for (auto tr = 0; tr < triangleBuff.size(); tr++)
    { // iterate triangles
        // vertices out of view frustum
        std::vector<uint32_t> out_buff{};

        for (auto i = 0; i < 3; i++)
        { // iterate vertices
            const auto& vx = triangleBuff[tr][i];
            // beyond clipping plane
            if (vx.gl_Position[2] < -vx.gl_Position[3])
                out_buff.push_back(i);
        }

        // checks N of vertices beyond clipping planes
        switch (out_buff.size())
        {
            case 1: // single beyond    --> two new triangles
                GPU::clip1(tr, out_buff);
                break;

            case 2: // two beyond       --> single new triangle
                GPU::clip2(triangleBuff[tr], out_buff);
                break;

            case 3: // all beyond       --> throw away
                triangleBuff.erase(triangleBuff.begin() + tr);
                tr--; // new triangle on same position
                break;

            default:
                break;
        }
    }
}

void GPU::clip1(uint32_t tr, std::vector<uint32_t>& outBuff)
{
    auto& triangle = triangleBuff[tr];

    // vertex beyond view plane -->tmpVx
    OutVertex vx(std::move(triangle[outBuff[0]]));
    triangle.erase(triangle.begin() + outBuff[0]);

    auto X1 = GPU::get_new_vertex(triangle[0], vx);
    auto X2 = GPU::get_new_vertex(triangle[1], vx);

    // 1st output triangle
    triangle.push_back(X1);
    // 2nd output triangle
    triangleBuff.push_back(Triangle{triangle[1], X2, X1});
}

void GPU::clip2(Triangle& triangle, std::vector<uint32_t>& out_buff)
{
    // get vertex that is inside
    OutVertex vx{};
    for (auto i = 0; i < 3; i++)
        if (i != out_buff[0] && i != out_buff[1])
        {
            vx = triangle[i];
            break;
        }

    triangle = {vx,
                GPU::get_new_vertex(triangle[out_buff[0]], vx),
                GPU::get_new_vertex(triangle[out_buff[1]], vx)};
}

OutVertex GPU::get_new_vertex(const OutVertex& A, const OutVertex& B)
{
    auto Az = A.gl_Position[2];
    auto Aw = A.gl_Position[3];
    auto Bz = B.gl_Position[2];
    auto Bw = B.gl_Position[3];
    auto D = (Bw - Aw + Bz - Az);
    auto t = (-Aw - Az) / D;

    OutVertex out{};

    glm::vec4 v = A.gl_Position + t * (B.gl_Position - A.gl_Position);
    for (auto i = 0; i < 4; i++) // ensures deep copy
        out.gl_Position[i] = v[i];

    // attribute interpolation
    glm::vec4 p;
    uint32_t size;
    for (auto i = 0; i < maxAttributes; i++)
    {
        size = sizeof(out.attributes[i]) / sizeof(float);
        p = {1 - t, t, 0, 0};
        GPU::interpol_attribs(out.attributes[i],
                              p,
                              i,
                              size,
                              A,
                              B,
                              B // this will be multiplied by 0
        );
    }

    return out;
}

void GPU::primAssembly(uint32_t N)
{
    if (N == 0)
        return;
    Triangle tr;
    const auto NVERTICES = 3; // each triangle == 3 vertices
    auto cnt = 0;

    for (auto i = 0; i < N; i++)
    {
        tr.push_back(GPU::vertexProcessor(i));

        // construct triangle
        if (++cnt == 3)
        {
            cnt = 0;
            triangleBuff.push_back(std::move(tr));
            if (i + 1 < N)
                tr.clear();
        }
    }
}

OutVertex GPU::vertexProcessor(uint32_t iteration)
{
    OutVertex out{};
    if (!isProgram(activePrg))
        return out;

    (*programs[activePrg]->VS)(
      out, GPU::vertexPuller(iteration), programs[activePrg]->uniforms);

    return out;
}

InVertex GPU::vertexPuller(uint32_t iteration)
{
    InVertex iv{};
    if (!isVertexPuller(activeVP))
        return iv;
    const auto vp = vertexPullers[activeVP].get();

    // Indexing
    if (vp->indexing.enabled) // get ID from index buffer
        getBufferData(vp->indexing.buffer,
                      iteration * (uint64_t)vp->indexing.type,
                      (uint64_t)vp->indexing.type,
                      &(iv.gl_VertexID));
    else
        iv.gl_VertexID = iteration;

    // Attributes
    const auto& heads = vp->heads;
    for (auto i = 0; i < heads.size(); i++)
    {
        if (heads[i]->enabled)
        {
            uint64_t offset =
              // offset           + stride           * ID
              heads[i]->offset + heads[i]->stride * iv.gl_VertexID;
            getBufferData(heads[i]->buffer,
                          offset,
                          (uint64_t)(heads[i]->type) * sizeof(float),
                          &(iv.attributes[i]));
        }
    }

    return iv;
}

/// @}
