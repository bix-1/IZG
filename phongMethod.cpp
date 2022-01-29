/*!
 * @file
 * @brief This file contains implementation of phong rendering method
 *
 * @author Tomáš Milet, imilet@fit.vutbr.cz
 */

#include <student/bunny.hpp>
#include <student/phongMethod.hpp>

/** \addtogroup shader_side 06. Implementace vertex/fragment shaderu phongovy
 * metody Vašim úkolem ve vertex a fragment shaderu je transformovat
 * trojúhelníky pomocí view a projekční matice a spočítat phongův osvětlovací
 * model. Vašim úkolem také je správně vypočítat procedurální barvu. Následující
 * obrázek zobrazuje shadery v různém stupni implementace. Horní řáděk zobrazuje
 * procedurální texturu. Prostřední řádek zobrazuje model králička s nanesenou
 * texturou, ale bez použití transformačních matic. Dolní řádek zobrazuje model
 * po použítí transformačních matic.
 *
 * \image html images/texture.svg "Vypočet procedurální textury." width=1000
 *
 *
 *
 *
 *
 * @{
 */

/**
 * @brief This function represents vertex shader of phong method.
 *
 * @param outVertex output vertex
 * @param inVertex input vertex
 * @param uniforms uniform variables
 */
void phong_VS(OutVertex& outVertex,
              InVertex const& inVertex,
              Uniforms const& uniforms)
{
    for (auto i = 0; i < 3; i++)
    {
        outVertex.attributes[0].v3[i] = inVertex.attributes[0].v3[i];
        outVertex.attributes[1].v3[i] = inVertex.attributes[1].v3[i];
    }
    outVertex.gl_Position = uniforms.uniform[1].m4 * uniforms.uniform[0].m4 *
                            inVertex.attributes[0].v4;
}

/**
 * @brief This function represents fragment shader of phong method.
 *
 * @param outFragment output fragment
 * @param inFragment input fragment
 * @param uniforms uniform variables
 */
void phong_FS(OutFragment& outFragment,
              InFragment const& inFragment,
              Uniforms const& uniforms)
{
    auto y = inFragment.attributes[1].v3[1]; // vNormal
    auto t = y * y;
    if (y > 0)
        outFragment.gl_FragColor = {t, t, t, 1};
    else
        outFragment.gl_FragColor = {-t, -t, -t, 1};

    auto tmp = inFragment.attributes[0].v4[0];              // x
    tmp += sin(inFragment.attributes[0].v4[1] * 10) / 10.f; // +sin curve
    tmp = fmod(tmp, 0.2f);
    if (tmp < 0)
        tmp += 0.2f; // fnc isn't even (instead of fabs())

    glm::vec4 GREEN = {0.f, 0.5f, 0.f, 0};
    glm::vec4 YELLOW = {1.f, 1.f, 0.f, 0};

    if (tmp < 0.1f) // green stripe
        outFragment.gl_FragColor += (1 - t) * GREEN;
    else // yellow stripe
        outFragment.gl_FragColor += (1 - t) * YELLOW;

    glm::vec3 vLight = uniforms.uniform[2].v3 - inFragment.attributes[0].v3;
    glm::vec3 vNormal = inFragment.attributes[1].v3;
    vLight = glm::normalize(vLight);
    vNormal = glm::normalize(vNormal);

    float LNdot = glm::dot(vNormal, vLight);
    LNdot = std::clamp(LNdot, 0.f, 1.f);
    // neg(camera pos)
    glm::vec3 viewDir = glm::normalize(-uniforms.uniform[3].v3);
    // R = I - 2( N * I )*N
    glm::vec3 reflectDir = glm::reflect(vLight, vNormal);
    float phongTerm = glm::dot(viewDir, reflectDir);
    phongTerm = std::clamp(phongTerm, 0.f, 1.f);
    // back-facing surface
    auto const EPS = 0.000001f;
    if (LNdot - EPS < 0.f)
        phongTerm = 0.f;
    auto shininessFactor = 40.f;
    phongTerm = pow(phongTerm, shininessFactor);

    glm::vec4 spColor = {1, 1, 1, 1};
    glm::vec4 I_light = {1, 1, 1, 1};

    outFragment.gl_FragColor =
      (outFragment.gl_FragColor * I_light * LNdot) + (spColor * phongTerm);

    // clamps to [ <0,1>, <0,1>, <0,1>, 1 ]
    outFragment.gl_FragColor[3] = 1.f;
    for (auto i = 0; i < 3; i++)
        outFragment.gl_FragColor[i] =
          std::clamp(outFragment.gl_FragColor[i], 0.f, 1.f);
}

/// @}

/** \addtogroup cpu_side 07. Implementace vykreslení králička s phongovým
 * osvětlovacím modelem. Vaším úkolem je využít naimplementovanou grafickou
 * kartu a vykreslit králička s phongovým osvětlovacím modelem a stínováním a
 * procedurální texturou.
 * @{
 */

/**
 * @brief Constructoro f phong method
 */
PhongMethod::PhongMethod()
{
    // Buffers
    // 1048 * float * 3 * 2 = 25 152
    auto vertexBuff = gpu.createBuffer(sizeof(bunnyVertices));
    auto indexBuff = gpu.createBuffer(sizeof(bunnyIndices));
    gpu.setBufferData(vertexBuff, 0, sizeof(bunnyVertices), bunnyVertices);
    gpu.setBufferData(indexBuff, 0, sizeof(bunnyIndices), bunnyIndices);

    // Vertex Puller
    vao = gpu.createVertexPuller();
    gpu.setVertexPullerHead(
      vao, 0, AttributeType::VEC3, sizeof(float) * 6, 0, vertexBuff);
    gpu.setVertexPullerHead(vao,
                            1,
                            AttributeType::VEC3,
                            6 * sizeof(float),
                            3 * sizeof(float),
                            vertexBuff);
    gpu.enableVertexPullerHead(vao, 0);
    gpu.enableVertexPullerHead(vao, 1);
    gpu.setVertexPullerIndexing(vao, IndexType::UINT32, indexBuff);

    prg = gpu.createProgram();
    gpu.attachShaders(prg, phong_VS, phong_FS);
    gpu.setVS2FSType(prg, 0, AttributeType::VEC3);
    gpu.setVS2FSType(prg, 1, AttributeType::VEC3);
}

/**
 * @brief This function draws phong method.
 *
 * @param proj projection matrix
 * @param view view matrix
 * @param light light position
 * @param camera camera position
 */
void PhongMethod::onDraw(glm::mat4 const& proj,
                         glm::mat4 const& view,
                         glm::vec3 const& light,
                         glm::vec3 const& camera)
{
    gpu.bindVertexPuller(vao);
    gpu.useProgram(prg);
    gpu.programUniformMatrix4f(prg, 0, view);
    gpu.programUniformMatrix4f(prg, 1, proj);
    gpu.programUniform3f(prg, 2, light);
    gpu.programUniform3f(prg, 3, camera);

    gpu.clear(.5f, .5f, .5f, 1.f);

    gpu.drawTriangles(nVERTICES);

    gpu.unbindVertexPuller();
}

/**
 * @brief Destructor of phong method.
 */
PhongMethod::~PhongMethod()
{
    // gpu.deleteProgram()
    // gpu.deleteVertexPuller()
    // gpu.deleteBuffer()
}

/// @}
