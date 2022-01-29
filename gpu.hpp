// gpu.hpp
// Project for course IZG @ FIT BUT
// Project Author:
//      Tomáš Milet, imilet@fit.vutbr.cz
// Implementation of Assignment:
//      Jakub Bartko, xbartk07
//      xbartk07@stud.fit.vutbr.cz
// FIT VUT Brno, 23.05.2020

// Description:
//    Header file of gpu.cpp

/*!
 * @file
 * @brief This file contains class that represents graphic card.
 *
 * @author Tomáš Milet, imilet@fit.vutbr.cz
 */
#pragma once

#include <cstring> // memcpy
#include <glm/gtx/string_cast.hpp>
#include <iostream> // cerr
#include <map>      // buffers
#include <memory>   // make_unique()
#include <numeric>  // glm::dot()
#include <student/fwd.hpp>
#include <vector> // Heads, VertexPullers

/**
 * @brief This class represent software GPU
 */
class GPU
{
public:
    GPU();
    virtual ~GPU();

    // buffer object commands
    BufferID createBuffer(uint64_t size);
    void deleteBuffer(BufferID buffer);
    void setBufferData(BufferID buffer,
                       uint64_t offset,
                       uint64_t size,
                       void const* data);
    void getBufferData(BufferID buffer,
                       uint64_t offset,
                       uint64_t size,
                       void* data);
    bool isBuffer(BufferID buffer);

    // vertex array object commands (vertex puller)
    ObjectID createVertexPuller();
    void deleteVertexPuller(VertexPullerID vao);
    void setVertexPullerHead(VertexPullerID vao,
                             uint32_t head,
                             AttributeType type,
                             uint64_t stride,
                             uint64_t offset,
                             BufferID buffer);
    void setVertexPullerIndexing(VertexPullerID vao,
                                 IndexType type,
                                 BufferID buffer);
    void enableVertexPullerHead(VertexPullerID vao, uint32_t head);
    void disableVertexPullerHead(VertexPullerID vao, uint32_t head);
    void bindVertexPuller(VertexPullerID vao);
    void unbindVertexPuller();
    bool isVertexPuller(VertexPullerID vao);

    // program object commands
    ProgramID createProgram();
    void deleteProgram(ProgramID prg);
    void attachShaders(ProgramID prg, VertexShader vs, FragmentShader fs);
    void setVS2FSType(ProgramID prg, uint32_t attrib, AttributeType type);
    void useProgram(ProgramID prg);
    bool isProgram(ProgramID prg);
    void programUniform1f(ProgramID prg, uint32_t uniformId, float const& d);
    void programUniform2f(ProgramID prg,
                          uint32_t uniformId,
                          glm::vec2 const& d);
    void programUniform3f(ProgramID prg,
                          uint32_t uniformId,
                          glm::vec3 const& d);
    void programUniform4f(ProgramID prg,
                          uint32_t uniformId,
                          glm::vec4 const& d);
    void programUniformMatrix4f(ProgramID prg,
                                uint32_t uniformId,
                                glm::mat4 const& d);

    // framebuffer functions
    void createFramebuffer(uint32_t width, uint32_t height);
    void deleteFramebuffer();
    void resizeFramebuffer(uint32_t width, uint32_t height);
    uint8_t* getFramebufferColor();
    float* getFramebufferDepth();
    uint32_t getFramebufferWidth();
    uint32_t getFramebufferHeight();

    // execution commands
    void clear(float r, float g, float b, float a);
    void drawTriangles(uint32_t nofVertices);

    /// \addtogroup gpu_init 00. proměnné, inicializace / deinicializace
    /// grafické karty
    /// @{
    /// \todo zde si můžete vytvořit proměnné grafické karty (buffery, programy,
    /// ...)
    /// @}
private:
    // *** *** *** Data Types *** *** ***
    struct Head
    {
        AttributeType type;
        uint64_t stride;
        uint64_t offset;
        BufferID buffer;
        bool enabled;
    };

    struct Indexing
    {
        bool enabled;
        BufferID buffer;
        IndexType type;
    };

    struct VertexPuller
    {
        Indexing indexing;
        std::vector<std::unique_ptr<Head>> heads;
    };

    struct Program
    {
        VertexShader VS;
        FragmentShader FS;
        Uniforms uniforms;
        std::vector<std::pair<uint32_t, AttributeType>> VStoFS;
    };

    typedef std::vector<OutVertex> Triangle;

    // *** *** *** Functions *** *** ***
    // pulls vertex from memory
    InVertex vertexPuller(uint32_t iteration);
    // runs vertexShader on given vertex
    OutVertex vertexProcessor(uint32_t iteration);
    // assembles given N of vertices into triangles in triangleBuff
    void primAssembly(uint32_t N);
    // clips triangles for near view plane
    void clipping();
    void clip1(uint32_t triangle, std::vector<uint32_t>& OutVertexBuff);
    void clip2(Triangle& triangle, std::vector<uint32_t>& OutVertexBuff);
    // returns vertex on view plane between A & B
    OutVertex get_new_vertex(const OutVertex&, const OutVertex&);
    // perspective division
    void persDivision();
    // viewport transformation
    void viewport_transf();
    // rasterization
    void rasterize();
    // returns bounds of triangle ( vector of [minX, maxX, minY, maxY] )
    std::vector<int> get_bounds(const OutVertex&,
                                const OutVertex&,
                                const OutVertex&);
    // interpolates attributes
    void interpol_attribs(Attribute& outAtt,
                          const glm::vec4& point,
                          const uint32_t pos,
                          const uint32_t size,
                          const OutVertex& inA,
                          const OutVertex& inB,
                          const OutVertex& inC);
    // baryc. coords of p for triangle ABC --> u,v,w
    void Barycentric(const glm::vec4 p,
                     const glm::vec4 A,
                     const glm::vec4 B,
                     const glm::vec4 C,
                     float& u,
                     float& v,
                     float& w);
    // runs Fragment Shader on given fragment
    OutFragment frag_proc(const InFragment&);
    // per fragment operations
    // fragments: inFragBuff --> FrameBuffer
    void perFrag_Op();

    // Global Variables
    VertexPullerID activeVP;
    ProgramID activePrg;
    struct
    {
        size_t width;
        size_t height;
        std::vector<uint8_t> ColorBuffer;
        std::vector<float> DepthBuffer;
    } FrameBuffer;

    // Global Buffers
    typedef std::vector<uint8_t> buffer_t;
    std::map<BufferID, std::unique_ptr<buffer_t>> buffers;
    std::map<VertexPullerID, std::unique_ptr<VertexPuller>> vertexPullers;
    std::map<ProgramID, std::unique_ptr<Program>> programs;
    std::vector<Triangle> triangleBuff;
    std::vector<InFragment> inFragBuff;
};
