// ============================================================================
//  3D Weather Visualization  -  C++ / OpenGL (fixed pipeline) / GLUT-FreeGLUT
//
//  A walk-around mountain lake scene with Sunny, Rainy, Snowy and Night modes.
//  Everything (terrain, textures, trees, cabin ...) is generated in code, so
//  no image or model files are needed.
//
//  HOW THE CODE FOLLOWS THE PROPOSED METHODOLOGY
//    Step 1  3D environment setup .... buildTerrain, placeTrees, placeRocks, placeGrass, drawCabin ...
//    Step 2  Rendering & transformation  glTranslatef / glRotatef / glScalef in every draw function,
//                                         lighting in setupLights, colours in terrainBaseColor
//    Step 3  Weather simulation ...... setMode (keys 1/2/3/4) and update(): sunny / rainy / snowy / night
//    Step 4  Particle animation ...... rain[] and snow[] in update() and drawParticles()
//    Step 5  Day and night ........... dayAmount, drawSky (sun, moon, stars), setupLights
//    Step 6  Testing ................. FPS counter on screen; all modes run continuously
//
//  Children play outside in sunny and snowy daytime weather; they go home when it rains or at night.
//
//  CONTROLS
//    1 / 2 / 3 / 4    Sunny / Rainy / Snowy / Night
//    W / S            Move forward / backward      (hold SHIFT to run)
//    A / D            Move left / right
//    Arrow keys       Look up / down, turn left / right  (mouse drag also works)
//    N                Toggle day / night for the current weather
//    P  pause    R  reset camera    H  hide help    F  fullscreen
//    Mouse wheel      Zoom (field of view)
//    ESC              Exit
// ============================================================================

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif
#ifdef __APPLE__
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#include <GLUT/glut.h>
#else
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <vector>
using std::vector;

// ----------------------------------------------------------------------------
//  Small math helpers
// ----------------------------------------------------------------------------
static const float PI = 3.14159265f;
struct V3  { float x, y, z; };
struct Col { float r, g, b; };

static inline float lerpf(float a, float b, float t) { return a + (b - a) * t; }
static inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
static inline float rad(float d) { return d * PI / 180.0f; }
static inline float smoothstep(float e0, float e1, float x) {
    float t = clampf((x - e0) / (e1 - e0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}
static inline Col mixc(Col a, Col b, float t) { return Col{lerpf(a.r, b.r, t), lerpf(a.g, b.g, t), lerpf(a.b, b.b, t)}; }
static inline Col scalec(Col a, float k) { return Col{a.r * k, a.g * k, a.b * k}; }
static inline V3 operator+(V3 a, V3 b) { return V3{a.x + b.x, a.y + b.y, a.z + b.z}; }
static inline V3 operator*(V3 a, float k) { return V3{a.x * k, a.y * k, a.z * k}; }
static inline float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 norm3(V3 a) { float l = sqrtf(dot3(a, a)); return l > 1e-6f ? a * (1.0f / l) : V3{0, 1, 0}; }

// exponential smoothing, independent of frame rate
static void approach(float &cur, float target, float dt, float rate) { cur += (target - cur) * (1.0f - expf(-rate * dt)); }
static void approachCol(Col &c, Col t, float dt, float rate) {
    approach(c.r, t.r, dt, rate); approach(c.g, t.g, dt, rate); approach(c.b, t.b, dt, rate);
}

// deterministic random numbers (same scene on every PC)
static unsigned int rs = 12345u;
static float frand() { rs = rs * 1664525u + 1013904223u; return ((rs >> 8) & 0xFFFFFF) / 16777216.0f; }
static float frand(float a, float b) { return a + (b - a) * frand(); }

// value noise / fbm / ridged noise
static unsigned int hash2(int x, int y) {
    unsigned int h = (unsigned)x * 374761393u + (unsigned)y * 668265263u;
    h = (h ^ (h >> 13)) * 1274126177u;
    return h ^ (h >> 16);
}
static float rnd2(int x, int y) { return (hash2(x, y) & 0xFFFFFFu) / 16777215.0f; }
static float vnoise(float x, float y) {
    int xi = (int)floorf(x), yi = (int)floorf(y);
    float fx = x - xi, fy = y - yi;
    float u = fx * fx * (3 - 2 * fx), v = fy * fy * (3 - 2 * fy);
    return lerpf(lerpf(rnd2(xi, yi), rnd2(xi + 1, yi), u), lerpf(rnd2(xi, yi + 1), rnd2(xi + 1, yi + 1), u), v);
}
static float fbm(float x, float y, int oct) {
    float s = 0, a = 0.5f, n = 0;
    for (int i = 0; i < oct; ++i) { s += a * vnoise(x, y); n += a; x = x * 2.03f + 17.0f; y = y * 2.03f + 5.0f; a *= 0.5f; }
    return s / n;
}
static float ridged(float x, float y, int oct) {
    float s = 0, a = 0.5f, n = 0;
    for (int i = 0; i < oct; ++i) {
        float r = 1.0f - fabsf(2.0f * vnoise(x, y) - 1.0f);
        s += r * r * a; n += a; x = x * 2.1f + 11.0f; y = y * 2.1f + 3.0f; a *= 0.5f;
    }
    return s / n;
}

// ----------------------------------------------------------------------------
//  World layout
// ----------------------------------------------------------------------------
static const float LAKE_CX = 15.0f, LAKE_CZ = -18.0f, LAKE_RX = 55.0f, LAKE_RZ = 34.0f;
static const float CABIN_X = -42.0f, CABIN_Z = 14.0f, CABIN_H = 2.6f, CABIN_YAW = 28.0f;

// terrain grid
static const int   TNX = 228, TNZ = 202;
static const float TCELL = 1.5f, TX0 = -170.0f, TZ0 = -190.0f;
static float tH[TNZ][TNX];

static float lakeDist(float x, float z) {
    float dx = (x - LAKE_CX) / LAKE_RX, dz = (z - LAKE_CZ) / LAKE_RZ;
    float d = sqrtf(dx * dx + dz * dz);
    return d * (0.88f + 0.28f * fbm(x * 0.045f + 31.0f, z * 0.045f + 17.0f, 3));
}

static float waterfallX(float z) { return 50.0f + 4.0f * sinf(z * 0.09f); }

static float terrainFunc(float x, float z) {
    float h = 2.4f + 3.2f * fbm(x * 0.018f + 10.0f, z * 0.018f + 3.0f, 4);       // rolling meadow
    float side = smoothstep(60.0f, 140.0f, fabsf(x - 10.0f));
    float mz = smoothstep(-48.0f, -130.0f, z);
    float m = fmaxf(mz, 0.5f * side);
    float rg = ridged(x * 0.010f + 3.0f, z * 0.010f + 9.0f, 5);
    float big = fbm(x * 0.006f + 5.0f, z * 0.006f - 2.0f, 3);
    h += side * (6.0f + 26.0f * fbm(x * 0.012f, z * 0.012f + 7.0f, 4));
    h += m * (10.0f + 105.0f * powf(rg, 1.55f) * (0.55f + 0.95f * big));          // mountains
    h += smoothstep(12.0f, 58.0f, z) * (1.0f - smoothstep(78.0f, 104.0f, z)) * 13.0f;   // slope up to the viewpoint
    float lf = 1.0f - smoothstep(0.65f, 1.12f, lakeDist(x, z));                    // lake basin
    h = h * (1.0f - lf) + (-6.0f) * lf;
    {   // waterfall channel: a steady slope from the lake up into the mountains
        float wz = clampf((-z - 44.0f) / 56.0f, 0.0f, 1.0f);
        float cw = (1.0f - smoothstep(2.5f, 10.0f, fabsf(x - waterfallX(z)))) * smoothstep(-42.0f, -48.0f, z);
        h = h * (1.0f - cw) + (-0.3f + 50.0f * powf(wz, 0.85f)) * cw;
    }
    float cf = 1.0f - smoothstep(7.0f, 16.0f, hypotf(x - CABIN_X, z - CABIN_Z));   // flat yard
    h = h * (1.0f - cf) + CABIN_H * cf;
    float e = fmaxf(fmaxf(smoothstep(150.0f, 168.0f, fabsf(x)), smoothstep(96.0f, 110.0f, z)),
                    smoothstep(-172.0f, -188.0f, z));                              // fade at borders
    return h * (1.0f - e) + 1.0f * e;
}

static float heightAt(float x, float z) {
    float fx = clampf((x - TX0) / TCELL, 0.0f, TNX - 1.001f), fz = clampf((z - TZ0) / TCELL, 0.0f, TNZ - 1.001f);
    int ix = (int)fx, iz = (int)fz;
    float u = fx - ix, v = fz - iz;
    return lerpf(lerpf(tH[iz][ix], tH[iz][ix + 1], u), lerpf(tH[iz + 1][ix], tH[iz + 1][ix + 1], u), v);
}
static V3 normalAt(float x, float z) {
    float e = 1.2f;
    return norm3(V3{heightAt(x - e, z) - heightAt(x + e, z), 2.0f * e, heightAt(x, z - e) - heightAt(x, z + e)});
}

// ----------------------------------------------------------------------------
//  Global state
// ----------------------------------------------------------------------------
static int   winW = 1280, winH = 720;
static int   weatherType = 0;                 // 0 clear, 1 rain, 2 snow
static bool  isDay = true, paused = false, showHelp = true, fullscreen = false;
static float timeSec = 0.0f, fpsShown = 0.0f;

// smoothly animated look values
static float dayAmount = 1.0f, snowCover = 0.0f, cloudCover = 0.3f, cloudDark = 0.0f;
static float rainAmt = 0.0f, snowAmt = 0.0f, sunVis = 1.0f, dimLight = 1.0f, chop = 0.35f;
static float fogS = 150.0f, fogE = 900.0f, flash = 0.0f, nextFlash = 6.0f;
static Col   skyTop = {0.15f, 0.38f, 0.82f}, skyHor = {0.60f, 0.78f, 0.95f};
static V3    sunDir = {0, 1, 0}, moonDir = {0, 1, 0};
static float lastSnowCover = -1.0f;
static float kidVis = 0.0f;                    // 1 = children are outside playing, 0 = gone home

// camera
static V3    camPos = {6.0f, 5.0f, 54.0f};
static float camYaw = -16.0f, camPitch = -2.0f, fovY = 58.0f;
static V3    camFwd = {0, 0, -1}, camRight = {1, 0, 0};
static bool  keyDown[256];
static bool  spLeft = false, spRight = false, spUp = false, spDown = false, shiftHeld = false;
static int   lastMX = 0, lastMY = 0;
static bool  dragging = false;

// textures
static GLuint texCloud = 0, texGlow = 0, texSun = 0, texMoon = 0;

// ----------------------------------------------------------------------------
//  Scene data
// ----------------------------------------------------------------------------
// terrain arrays (vertex arrays; colours are refreshed when snow cover changes)
static vector<float> vPos, vNrm, vCol, baseCol, peakSnow, vNy;
static vector<unsigned int> vIdx;

struct Tree  { float x, y, z, h, w, v; int type; };
struct Rock  { float x, y, z, sx, sy, sz, rot, shade; };
struct Tuft  { float x, y, z, h, rot; Col c; };
struct Flower{ float x, y, z; Col c; };
struct Puff  { float x, y, z, size, thr, tone; };
struct Drop  { float x, y, z, speed, drift, freq, phase; };
struct Ripple{ float x, z, age, life; };
struct Bird  { float cx, cz, r, y, speed, phase, flap; };

static vector<Tree>   trees;
static vector<Rock>   rocks;
static vector<Tuft>   tufts;
static vector<Flower> flowers;
static vector<Puff>   puffs;
static vector<Drop>   rain, snow;
static vector<Ripple> ripples;
static vector<Bird>   birds;
static vector<V3>     fencePosts;
static vector<V3>     fallPath;                // waterfall centre line
static vector<float>  stars;
static vector<int>    visTrees;

static GLuint dlTrunk, dlPineA, dlPineB, dlPineFar, dlBall;

static const int NUM_RAIN = 7000, NUM_SNOW = 5000;
static const float PBOX = 38.0f, PTOP = 36.0f;

// ----------------------------------------------------------------------------
//  Procedural geometry (display lists)
// ----------------------------------------------------------------------------
static void coneGeom(float y0, float r, float h, int sl) {
    float len = sqrtf(h * h + r * r);
    glBegin(GL_TRIANGLES);
    for (int i = 0; i < sl; ++i) {
        float a0 = 2 * PI * i / sl, a1 = 2 * PI * (i + 1) / sl, am = 0.5f * (a0 + a1);
        glNormal3f(cosf(am) * h / len, r / len, sinf(am) * h / len);
        glVertex3f(0, y0 + h, 0);
        glNormal3f(cosf(a1) * h / len, r / len, sinf(a1) * h / len);
        glVertex3f(cosf(a1) * r, y0, sinf(a1) * r);
        glNormal3f(cosf(a0) * h / len, r / len, sinf(a0) * h / len);
        glVertex3f(cosf(a0) * r, y0, sinf(a0) * r);
    }
    glEnd();
}
static void cylGeom(float r, float h, int sl) {
    glBegin(GL_QUADS);
    for (int i = 0; i < sl; ++i) {
        float a0 = 2 * PI * i / sl, a1 = 2 * PI * (i + 1) / sl;
        glNormal3f(cosf(a0), 0, sinf(a0)); glVertex3f(cosf(a0) * r, 0, sinf(a0) * r); glVertex3f(cosf(a0) * r, h, sinf(a0) * r);
        glNormal3f(cosf(a1), 0, sinf(a1)); glVertex3f(cosf(a1) * r, h, sinf(a1) * r); glVertex3f(cosf(a1) * r, 0, sinf(a1) * r);
    }
    glEnd();
}
static void buildLists() {
    dlTrunk = glGenLists(5);
    glNewList(dlTrunk, GL_COMPILE); cylGeom(0.035f, 0.28f, 6); glEndList();
    dlPineA = dlTrunk + 1;
    glNewList(dlPineA, GL_COMPILE); coneGeom(0.12f, 0.36f, 0.44f, 10); coneGeom(0.34f, 0.29f, 0.38f, 10); glEndList();
    dlPineB = dlTrunk + 2;
    glNewList(dlPineB, GL_COMPILE); coneGeom(0.54f, 0.21f, 0.34f, 9); coneGeom(0.72f, 0.12f, 0.28f, 8); glEndList();
    dlPineFar = dlTrunk + 3;
    glNewList(dlPineFar, GL_COMPILE); coneGeom(0.08f, 0.32f, 0.92f, 6); glEndList();
    dlBall = dlTrunk + 4;
    glNewList(dlBall, GL_COMPILE); glutSolidSphere(1.0, 9, 7); glEndList();
}

// ----------------------------------------------------------------------------
//  Textures (all generated in code)
// ----------------------------------------------------------------------------
static GLuint makeTex(int n, const vector<unsigned char> &px) {
    GLuint t; glGenTextures(1, &t); glBindTexture(GL_TEXTURE_2D, t);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    gluBuild2DMipmaps(GL_TEXTURE_2D, 4, n, n, GL_RGBA, GL_UNSIGNED_BYTE, &px[0]);
    return t;
}
static void buildTextures() {
    const int N = 128;
    vector<unsigned char> px(N * N * 4);
    for (int y = 0; y < N; ++y) for (int x = 0; x < N; ++x) {          // soft cloud puff
        float u = (x / (N - 1.0f) - 0.5f) * 2, v = (y / (N - 1.0f) - 0.5f) * 2, r = sqrtf(u * u + v * v);
        float a = (1.0f - smoothstep(0.18f, 1.0f, r)) * (0.30f + 1.25f * fbm(x * 0.055f + 40.0f, y * 0.055f, 4));
        a = powf(clampf(a, 0, 1), 1.25f);
        float lum = 0.74f + 0.26f * (y / (N - 1.0f));
        unsigned char *p = &px[(y * N + x) * 4];
        p[0] = p[1] = (unsigned char)(lum * 255); p[2] = (unsigned char)(clampf(lum + 0.03f, 0, 1) * 255); p[3] = (unsigned char)(a * 255);
    }
    texCloud = makeTex(N, px);
    for (int y = 0; y < N; ++y) for (int x = 0; x < N; ++x) {          // glow
        float u = (x / (N - 1.0f) - 0.5f) * 2, v = (y / (N - 1.0f) - 0.5f) * 2, r = sqrtf(u * u + v * v);
        float a = powf(clampf(1.0f - r, 0, 1), 2.2f);
        unsigned char *p = &px[(y * N + x) * 4]; p[0] = p[1] = p[2] = 255; p[3] = (unsigned char)(a * 255);
    }
    texGlow = makeTex(N, px);
    for (int y = 0; y < N; ++y) for (int x = 0; x < N; ++x) {          // sun disc
        float u = (x / (N - 1.0f) - 0.5f) * 2, v = (y / (N - 1.0f) - 0.5f) * 2, r = sqrtf(u * u + v * v);
        unsigned char *p = &px[(y * N + x) * 4]; p[0] = p[1] = p[2] = 255; p[3] = (unsigned char)((1.0f - smoothstep(0.78f, 1.0f, r)) * 255);
    }
    texSun = makeTex(N, px);
    for (int y = 0; y < N; ++y) for (int x = 0; x < N; ++x) {          // moon with craters
        float u = (x / (N - 1.0f) - 0.5f) * 2, v = (y / (N - 1.0f) - 0.5f) * 2, r = sqrtf(u * u + v * v);
        float n = fbm(x * 0.07f + 90.0f, y * 0.07f + 20.0f, 4);
        float c = 0.93f - 0.30f * smoothstep(0.50f, 0.68f, n) - 0.06f * vnoise(x * 0.4f, y * 0.4f);
        unsigned char *p = &px[(y * N + x) * 4];
        p[0] = (unsigned char)(c * 255); p[1] = (unsigned char)(c * 252); p[2] = (unsigned char)(c * 238);
        p[3] = (unsigned char)((1.0f - smoothstep(0.90f, 1.0f, r)) * 255);
    }
    texMoon = makeTex(N, px);
}

// ----------------------------------------------------------------------------
//  Scene construction
// ----------------------------------------------------------------------------
static Col terrainBaseColor(float x, float z, float h, float ny, float &peak) {
    float n1 = fbm(x * 0.08f, z * 0.08f, 3), n2 = vnoise(x * 0.55f, z * 0.55f);
    Col grassA = {0.26f, 0.50f, 0.12f}, grassB = {0.42f, 0.62f, 0.17f}, forest = {0.12f, 0.29f, 0.09f};
    Col rock = {0.45f, 0.42f, 0.39f}, dirt = {0.34f, 0.28f, 0.19f}, alpine = {0.46f, 0.51f, 0.31f}, bed = {0.10f, 0.15f, 0.14f};
    Col c = mixc(grassA, grassB, n1);
    c = mixc(c, forest, smoothstep(0.46f, 0.62f, fbm(x * 0.035f + 50.0f, z * 0.035f, 3)) * 0.8f);
    c = scalec(c, 0.88f + 0.24f * n2);
    c = mixc(c, dirt, 1.0f - smoothstep(0.0f, 1.4f, h));
    c = mixc(c, alpine, smoothstep(24.0f, 44.0f, h) * 0.8f);
    float rockF = clampf(smoothstep(0.78f, 0.56f, ny) + smoothstep(44.0f, 66.0f, h) * 0.7f, 0, 1);
    c = mixc(c, scalec(rock, 0.8f + 0.4f * n2), rockF);
    c = mixc(c, bed, 1.0f - smoothstep(-2.5f, 0.0f, h));
    float line = 38.0f + 12.0f * (fbm(x * 0.03f, z * 0.03f, 3) - 0.5f);
    peak = smoothstep(line, line + 9.0f, h) * smoothstep(0.38f, 0.70f, ny + 0.3f * (n1 - 0.5f));
    return c;
}

static void buildTerrain() {
    for (int j = 0; j < TNZ; ++j) for (int i = 0; i < TNX; ++i) tH[j][i] = terrainFunc(TX0 + i * TCELL, TZ0 + j * TCELL);
    int N = TNX * TNZ;
    vPos.resize(N * 3); vNrm.resize(N * 3); vCol.resize(N * 3); baseCol.resize(N * 3); peakSnow.resize(N); vNy.resize(N);
    for (int j = 0; j < TNZ; ++j) for (int i = 0; i < TNX; ++i) {
        int k = j * TNX + i;
        float x = TX0 + i * TCELL, z = TZ0 + j * TCELL, h = tH[j][i];
        float hl = tH[j][i > 0 ? i - 1 : i], hr = tH[j][i < TNX - 1 ? i + 1 : i];
        float hd = tH[j > 0 ? j - 1 : j][i], hu = tH[j < TNZ - 1 ? j + 1 : j][i];
        V3 n = norm3(V3{hl - hr, 2 * TCELL, hd - hu});
        vPos[k * 3] = x; vPos[k * 3 + 1] = h; vPos[k * 3 + 2] = z;
        vNrm[k * 3] = n.x; vNrm[k * 3 + 1] = n.y; vNrm[k * 3 + 2] = n.z; vNy[k] = n.y;
        float pk; Col c = terrainBaseColor(x, z, h, n.y, pk);
        baseCol[k * 3] = c.r; baseCol[k * 3 + 1] = c.g; baseCol[k * 3 + 2] = c.b; peakSnow[k] = pk;
    }
    for (int j = 0; j < TNZ - 1; ++j) for (int i = 0; i < TNX - 1; ++i) {
        unsigned int a = j * TNX + i, b = a + 1, c = a + TNX, d = c + 1;
        vIdx.push_back(a); vIdx.push_back(c); vIdx.push_back(b);
        vIdx.push_back(b); vIdx.push_back(c); vIdx.push_back(d);
    }
}

static void refreshTerrainColors() {
    int N = TNX * TNZ;
    const Col snow = {0.93f, 0.95f, 0.99f};
    for (int k = 0; k < N; ++k) {
        float ny = vNy[k], h = vPos[k * 3 + 1];
        float cover = snowCover * smoothstep(0.30f, 0.62f, ny) * smoothstep(-0.5f, 0.3f, h);
        float s = clampf(fmaxf(peakSnow[k], cover), 0.0f, 1.0f);
        vCol[k * 3]     = lerpf(baseCol[k * 3],     snow.r, s);
        vCol[k * 3 + 1] = lerpf(baseCol[k * 3 + 1], snow.g, s);
        vCol[k * 3 + 2] = lerpf(baseCol[k * 3 + 2], snow.b, s);
    }
    lastSnowCover = snowCover;
}

static void placeFence() {
    float len = 0, px = 0, pz = 0; bool first = true;
    for (int i = 0; i <= 400; ++i) {
        float t = i / 400.0f;
        float x = -62.0f + 54.0f * t, z = 20.0f + 24.0f * t + 5.0f * sinf(t * 3.2f);
        if (!first) len += hypotf(x - px, z - pz);
        if (first || len >= 3.0f) { fencePosts.push_back(V3{x, heightAt(x, z), z}); len = 0; }
        first = false; px = x; pz = z;
    }
}

static bool nearFence(float x, float z, float r) {
    for (size_t i = 0; i < fencePosts.size(); ++i) if (hypotf(x - fencePosts[i].x, z - fencePosts[i].z) < r) return true;
    return false;
}

static void placeTrees() {
    const int GW = 120, GH = 100;
    vector<char> occ(GW * GH, 0);
    int tries = 0;
    while (trees.size() < 2300 && tries < 150000) {
        ++tries;
        float x = frand(-165, 165), z = frand(-180, 104), h = heightAt(x, z);
        if (h < 1.2f || h > 38.0f) continue;
        if (normalAt(x, z).y < 0.80f) continue;
        if (hypotf(x - CABIN_X, z - CABIN_Z) < 15.0f || nearFence(x, z, 4.0f)) continue;
        if (z < -40.0f && fabsf(x - waterfallX(z)) < 7.0f) continue;
        float dens = 0.05f + 0.95f * smoothstep(0.46f, 0.60f, fbm(x * 0.035f + 50.0f, z * 0.035f, 3));
        if (hypotf(x - 6.0f, z - 54.0f) < 28.0f) continue;               // clear meadow around the start point
        dens *= 0.04f + 0.96f * smoothstep(1.15f, 1.9f, lakeDist(x, z)); // open shoreline so the lake is visible
        if (z > 30.0f && fabsf(x - 5.0f) < 75.0f) dens *= 0.15f;
        if (x > -34.0f && x < 85.0f && z > -6.0f && z < 52.0f) dens *= 0.02f;   // open view from the viewpoint to the lake
        if (frand() > dens) continue;
        int cx = (int)((x - TX0) / 3.0f), cz = (int)((z - TZ0) / 3.0f);
        if (cx < 0 || cz < 0 || cx >= GW || cz >= GH || occ[cz * GW + cx]) continue;
        occ[cz * GW + cx] = 1;
        Tree t; t.x = x; t.y = h - 0.15f; t.z = z; t.v = frand(0.75f, 1.2f);
        bool decid = (h < 10.0f && frand() < 0.16f);
        t.type = decid ? 1 : 0;
        if (decid) { t.h = frand(5.0f, 8.0f); t.w = t.h * 0.62f; }
        else { t.h = frand(7.0f, 16.0f) * (1.0f - 0.5f * smoothstep(26.0f, 38.0f, h)); t.w = t.h * 0.6f * frand(0.85f, 1.1f); }
        trees.push_back(t);
    }
}

static void placeRocks() {
    int tries = 0;
    while (rocks.size() < 55 && tries++ < 20000) {
        float x = frand(-110, 110), z = frand(-60, 90), h = heightAt(x, z), d = lakeDist(x, z);
        bool shore = d > 0.85f && d < 1.08f;
        if (!(shore || frand() < 0.03f) || h < -0.3f || h > 14.0f) continue;
        if (hypotf(x - CABIN_X, z - CABIN_Z) < 12.0f || nearFence(x, z, 3.0f)) continue;
        float s = frand(0.5f, 2.0f);
        rocks.push_back(Rock{x, h, z, s * frand(0.9f, 1.5f), s * frand(0.55f, 0.9f), s * frand(0.9f, 1.4f), frand(0, 360), frand(0.8f, 1.15f)});
    }
}

static void placeGrass() {
    int tries = 0;
    while (tufts.size() < 9000 && tries++ < 90000) {
        float x = frand(-110, 110), z = frand(-30, 98), h = heightAt(x, z);
        if (h < 0.8f || h > 14.0f || normalAt(x, z).y < 0.88f) continue;
        if (hypotf(x - CABIN_X, z - CABIN_Z) < 4.0f) continue;
        Tuft t; t.x = x; t.y = h; t.z = z; t.h = frand(0.22f, 0.5f); t.rot = frand(0, 6.28f);
        float v = frand();
        t.c = mixc(Col{0.22f, 0.46f, 0.10f}, Col{0.50f, 0.68f, 0.20f}, v);
        tufts.push_back(t);
    }
    tries = 0;
    while (flowers.size() < 900 && tries++ < 60000) {
        float x = frand(-90, 90), z = frand(-20, 96), h = heightAt(x, z);
        if (h < 1.0f || h > 10.0f || normalAt(x, z).y < 0.9f) continue;
        if (fbm(x * 0.06f + 3.0f, z * 0.06f + 1.0f, 3) < 0.50f) continue;             // flowers grow in patches
        float k = frand();
        Col c = k < 0.4f ? Col{1.0f, 1.0f, 0.95f} : (k < 0.7f ? Col{1.0f, 0.86f, 0.25f} : (k < 0.88f ? Col{0.75f, 0.45f, 0.95f} : Col{1.0f, 0.55f, 0.65f}));
        flowers.push_back(Flower{x, h + 0.45f, z, c});
    }
}

static void placeSky() {
    for (int i = 0; i < 520; ++i) {                                    // stars
        float th = frand(0, 2 * PI), el = frand(0.05f, 1.5f), r = 470.0f;
        stars.push_back(r * cosf(el) * cosf(th)); stars.push_back(r * sinf(el)); stars.push_back(r * cosf(el) * sinf(th));
    }
    const int NC = 56;
    for (int c = 0; c < NC; ++c) {                                     // clouds: clusters of soft puffs
        float a = frand(0, 2 * PI), r = frand(150, 650), cx = r * cosf(a), cz = r * sinf(a) - 80.0f, cy = frand(150, 260);
        float thr = (c + 0.5f) / NC, tone = frand(0.9f, 1.0f);
        int np = 5 + (int)frand(0, 4);
        for (int k = 0; k < np; ++k)
            puffs.push_back(Puff{cx + frand(-90, 90), cy + frand(-18, 18), cz + frand(-60, 60), frand(95, 190), thr, tone});
    }
    for (int i = 0; i < NUM_RAIN; ++i) rain.push_back(Drop{frand(-PBOX, PBOX), frand(0, PTOP), frand(-PBOX, PBOX), frand(30, 40), 0, 0, 0});
    for (int i = 0; i < NUM_SNOW; ++i)
        snow.push_back(Drop{frand(-PBOX, PBOX), frand(0, PTOP), frand(-PBOX, PBOX), frand(1.4f, 3.2f), frand(0.4f, 1.5f), frand(0.6f, 1.8f), frand(0, 2 * PI)});
    for (int i = 0; i < 7; ++i)
        birds.push_back(Bird{frand(-10, 40), frand(-30, 10), frand(18, 45), frand(24, 42), frand(0.25f, 0.45f), frand(0, 6.28f), frand(7, 11)});
}

static void placeWaterfall() {
    for (float z = -102.0f; z <= -44.0f; z += 1.5f) {
        float x = waterfallX(z);
        fallPath.push_back(V3{x, heightAt(x, z), z});
    }
}

static void initKids();     // defined further below

static void initScene() {
    buildTerrain();
    placeFence();
    placeTrees();
    placeRocks();
    placeGrass();
    placeSky();
    placeWaterfall();
    initKids();
    for (int i = 0; i < 60; ++i) ripples.push_back(Ripple{0, 0, 9.0f, 1.0f});
    refreshTerrainColors();
}

// ----------------------------------------------------------------------------
//  Update (animation + smooth transitions)
// ----------------------------------------------------------------------------
static const Col DAY_TOP[3] = {{0.15f, 0.38f, 0.82f}, {0.27f, 0.30f, 0.35f}, {0.52f, 0.60f, 0.70f}};
static const Col DAY_HOR[3] = {{0.60f, 0.78f, 0.95f}, {0.46f, 0.50f, 0.55f}, {0.78f, 0.84f, 0.90f}};
static const Col NGT_TOP[3] = {{0.015f, 0.03f, 0.11f}, {0.03f, 0.04f, 0.07f}, {0.07f, 0.09f, 0.17f}};
static const Col NGT_HOR[3] = {{0.07f, 0.11f, 0.25f}, {0.06f, 0.07f, 0.11f}, {0.17f, 0.21f, 0.31f}};

static void setMode(int w, bool day) { weatherType = w; isDay = day; }

static void updateCamera(float dt) {
    float run = shiftHeld ? 2.6f : 1.0f, sp = 14.0f * run * dt;
    float yaw = rad(camYaw), fx = sinf(yaw), fz = -cosf(yaw), rx = cosf(yaw), rz = sinf(yaw);
    float mx = 0, mz = 0;
    if (keyDown['w']) { mx += fx; mz += fz; }
    if (keyDown['s']) { mx -= fx; mz -= fz; }
    if (keyDown['d']) { mx += rx; mz += rz; }
    if (keyDown['a']) { mx -= rx; mz -= rz; }
    camPos.x = clampf(camPos.x + mx * sp, -125.0f, 125.0f);
    camPos.z = clampf(camPos.z + mz * sp, -35.0f, 88.0f);
    if (spLeft)  camYaw -= 70.0f * dt;
    if (spRight) camYaw += 70.0f * dt;
    if (spUp)    camPitch += 45.0f * dt;
    if (spDown)  camPitch -= 45.0f * dt;
    camPitch = clampf(camPitch, -60.0f, 75.0f);
    float want = fmaxf(heightAt(camPos.x, camPos.z), 0.0f) + 1.9f;     // walk on the ground, hover over water
    approach(camPos.y, want, dt, 10.0f);
}

static void update(float dt) {
    static const float tCover[3] = {0.30f, 1.00f, 0.88f}, tDark[3] = {0.0f, 1.0f, 0.30f}, tSun[3] = {1.0f, 0.0f, 0.45f};
    static const float tDim[3] = {1.0f, 0.42f, 0.62f}, tChop[3] = {0.35f, 1.0f, 0.5f};
    static const float tFogS[3] = {150.0f, 18.0f, 25.0f}, tFogE[3] = {900.0f, 300.0f, 420.0f};
    int w = weatherType;
    float nk = isDay ? 1.0f : 0.7f;
    approach(snowCover, w == 2 ? 1.0f : 0.0f, dt, 0.45f);
    approach(cloudCover, tCover[w], dt, 1.0f);  approach(cloudDark, tDark[w], dt, 1.2f);
    approach(rainAmt, w == 1 ? 1.0f : 0.0f, dt, 1.6f);  approach(snowAmt, w == 2 ? 1.0f : 0.0f, dt, 1.6f);
    approach(sunVis, tSun[w], dt, 1.5f);  approach(dimLight, tDim[w], dt, 1.5f);  approach(chop, tChop[w], dt, 1.0f);
    approach(fogS, tFogS[w] * nk, dt, 1.2f);  approach(fogE, tFogE[w] * nk, dt, 1.2f);
    approach(kidVis, (isDay && weatherType != 1) ? 1.0f : 0.0f, dt, 2.2f);   // children go home in rain and at night
    float step = dt * 0.45f, tgt = isDay ? 1.0f : 0.0f;
    dayAmount += clampf(tgt - dayAmount, -step, step);

    approachCol(skyTop, mixc(NGT_TOP[w], DAY_TOP[w], dayAmount), dt, 3.0f);
    approachCol(skyHor, mixc(NGT_HOR[w], DAY_HOR[w], dayAmount), dt, 3.0f);

    float sunEl = rad(lerpf(-20.0f, 24.0f, dayAmount)), moonEl = rad(lerpf(21.0f, -20.0f, dayAmount));
    float sa = rad(-8.0f), ma = rad(-5.0f);
    sunDir  = V3{cosf(sunEl) * sinf(sa),  sinf(sunEl),  -cosf(sunEl) * cosf(sa)};
    moonDir = V3{cosf(moonEl) * sinf(ma), sinf(moonEl), -cosf(moonEl) * cosf(ma)};

    if (fabsf(snowCover - lastSnowCover) > 0.004f) refreshTerrainColors();
    if (paused) return;
    timeSec += dt;

    // lightning during rain
    if (rainAmt > 0.6f) { nextFlash -= dt; if (nextFlash <= 0) { flash = 1.0f; nextFlash = frand(7.0f, 16.0f); } }
    flash = fmaxf(0.0f, flash - dt * 2.8f);

    // rain: fast and straight, wraps around the camera
    for (size_t i = 0; i < rain.size(); ++i) {
        Drop &p = rain[i];
        p.y -= p.speed * dt;
        if (p.y < 0) { p.y += PTOP; p.x = camPos.x + frand(-PBOX, PBOX); p.z = camPos.z + frand(-PBOX, PBOX); }
        if (p.x - camPos.x > PBOX) p.x -= 2 * PBOX; else if (p.x - camPos.x < -PBOX) p.x += 2 * PBOX;
        if (p.z - camPos.z > PBOX) p.z -= 2 * PBOX; else if (p.z - camPos.z < -PBOX) p.z += 2 * PBOX;
    }
    // snow: slow, with sideways drift
    for (size_t i = 0; i < snow.size(); ++i) {
        Drop &p = snow[i];
        p.y -= p.speed * dt;
        p.x += sinf(timeSec * p.freq + p.phase) * p.drift * dt;
        p.z += cosf(timeSec * p.freq * 0.8f + p.phase) * p.drift * 0.6f * dt;
        if (p.y < 0) { p.y += PTOP; p.x = camPos.x + frand(-PBOX, PBOX); p.z = camPos.z + frand(-PBOX, PBOX); }
        if (p.x - camPos.x > PBOX) p.x -= 2 * PBOX; else if (p.x - camPos.x < -PBOX) p.x += 2 * PBOX;
        if (p.z - camPos.z > PBOX) p.z -= 2 * PBOX; else if (p.z - camPos.z < -PBOX) p.z += 2 * PBOX;
    }
    for (size_t i = 0; i < puffs.size(); ++i) { puffs[i].x += (1.2f + 0.02f * puffs[i].size) * dt * (0.5f + 0.5f * chop); if (puffs[i].x > 700) puffs[i].x = -700; }
    for (size_t i = 0; i < birds.size(); ++i) birds[i].phase += birds[i].speed * dt;
    for (size_t i = 0; i < ripples.size(); ++i) {
        Ripple &r = ripples[i]; r.age += dt;
        if (r.age > r.life && rainAmt > 0.5f) {
            for (int k = 0; k < 12; ++k) {
                float x = frand(-40, 80), z = frand(-55, 20);
                if (heightAt(x, z) < -0.8f) { r.x = x; r.z = z; r.age = 0; r.life = frand(0.7f, 1.4f); break; }
            }
        }
    }
}

// ----------------------------------------------------------------------------
//  Lighting
// ----------------------------------------------------------------------------
static void setupLights() {
    glEnable(GL_LIGHTING); glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glEnable(GL_NORMALIZE);
    glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_TRUE);

    float fl = flash * (0.6f + 0.4f * sinf(timeSec * 60.0f));
    Col a = mixc(Col{0.24f, 0.29f, 0.48f}, Col{0.42f, 0.47f, 0.56f}, dayAmount);
    float ak = 0.55f + 0.45f * dimLight + 0.12f * snowAmt;
    float amb[4] = {a.r * ak + fl * 0.5f, a.g * ak + fl * 0.5f, a.b * ak + fl * 0.55f, 1};
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, amb);

    float zero[4] = {0, 0, 0, 1};
    float k = dayAmount * dimLight * 1.12f + fl * 0.6f;
    float sd[4] = {1.0f * k, 0.88f * k, 0.66f * k, 1}, sp[4] = {sd[0] * 0.9f, sd[1] * 0.9f, sd[2] * 0.9f, 1};
    float spos[4] = {sunDir.x, sunDir.y, sunDir.z, 0};
    glEnable(GL_LIGHT0);
    glLightfv(GL_LIGHT0, GL_AMBIENT, zero); glLightfv(GL_LIGHT0, GL_DIFFUSE, sd); glLightfv(GL_LIGHT0, GL_SPECULAR, sp); glLightfv(GL_LIGHT0, GL_POSITION, spos);

    float mk = (1.0f - dayAmount) * (weatherType == 1 ? 0.55f : 1.0f);
    float md[4] = {0.70f * mk, 0.86f * mk, 1.25f * mk, 1}, ms[4] = {md[0] * 1.5f, md[1] * 1.5f, md[2] * 1.5f, 1};
    float mpos[4] = {moonDir.x, moonDir.y, moonDir.z, 0};
    glEnable(GL_LIGHT1);
    glLightfv(GL_LIGHT1, GL_AMBIENT, zero); glLightfv(GL_LIGHT1, GL_DIFFUSE, md); glLightfv(GL_LIGHT1, GL_SPECULAR, ms); glLightfv(GL_LIGHT1, GL_POSITION, mpos);

    float fk = dayAmount * (0.30f + 0.20f * dimLight);                  // soft fill light from the viewer's side
    float fd[4] = {0.85f * fk, 0.92f * fk, 1.0f * fk, 1}, fpos[4] = {-camFwd.x, 0.55f, -camFwd.z, 0};
    glEnable(GL_LIGHT3);
    glLightfv(GL_LIGHT3, GL_AMBIENT, zero); glLightfv(GL_LIGHT3, GL_DIFFUSE, fd); glLightfv(GL_LIGHT3, GL_SPECULAR, zero); glLightfv(GL_LIGHT3, GL_POSITION, fpos);

    float lk = (1.0f - dayAmount) * 1.3f;                              // warm lamp at the cabin
    float ld[4] = {1.0f * lk, 0.68f * lk, 0.34f * lk, 1};
    float lpos[4] = {CABIN_X + 3.0f, CABIN_H + 3.0f, CABIN_Z + 7.0f, 1};
    glEnable(GL_LIGHT2);
    glLightfv(GL_LIGHT2, GL_AMBIENT, zero); glLightfv(GL_LIGHT2, GL_DIFFUSE, ld); glLightfv(GL_LIGHT2, GL_SPECULAR, zero); glLightfv(GL_LIGHT2, GL_POSITION, lpos);
    glLightf(GL_LIGHT2, GL_CONSTANT_ATTENUATION, 1.0f); glLightf(GL_LIGHT2, GL_LINEAR_ATTENUATION, 0.03f); glLightf(GL_LIGHT2, GL_QUADRATIC_ATTENUATION, 0.006f);
}

static void setFog() {
    float fc[4] = {skyHor.r, skyHor.g, skyHor.b, 1};
    glEnable(GL_FOG); glFogi(GL_FOG_MODE, GL_LINEAR);
    glFogfv(GL_FOG_COLOR, fc); glFogf(GL_FOG_START, fogS); glFogf(GL_FOG_END, fogE);
}

// ----------------------------------------------------------------------------
//  Sky
// ----------------------------------------------------------------------------
static V3 bbRight, bbUp;
static void beginBillboards() {
    float m[16]; glGetFloatv(GL_MODELVIEW_MATRIX, m);
    bbRight = V3{m[0], m[4], m[8]}; bbUp = V3{m[1], m[5], m[9]};
}
static void sprite(V3 c, float size) {
    V3 r = bbRight * size, u = bbUp * size;
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0); glVertex3f(c.x - r.x - u.x, c.y - r.y - u.y, c.z - r.z - u.z);
    glTexCoord2f(1, 0); glVertex3f(c.x + r.x - u.x, c.y + r.y - u.y, c.z + r.z - u.z);
    glTexCoord2f(1, 1); glVertex3f(c.x + r.x + u.x, c.y + r.y + u.y, c.z + r.z + u.z);
    glTexCoord2f(0, 1); glVertex3f(c.x - r.x + u.x, c.y - r.y + u.y, c.z - r.z + u.z);
    glEnd();
}

static void drawSky() {
    glDisable(GL_LIGHTING); glDisable(GL_FOG); glDisable(GL_DEPTH_TEST); glDepthMask(GL_FALSE); glDisable(GL_TEXTURE_2D);
    glPushMatrix(); glTranslatef(camPos.x, camPos.y, camPos.z);
    const int ST = 20, SL = 40; const float R = 500.0f;
    float fl = flash * (0.6f + 0.4f * sinf(timeSec * 60.0f));
    for (int i = 0; i < ST; ++i) {
        glBegin(GL_QUAD_STRIP);
        for (int j = 0; j <= SL; ++j) for (int s = 1; s >= 0; --s) {
            float e = rad(-25.0f + 115.0f * (i + s) / ST), a = 2 * PI * j / SL;
            V3 d = {cosf(e) * cosf(a), sinf(e), cosf(e) * sinf(a)};
            float t = powf(clampf(d.y, 0, 1), 0.55f);
            Col c = mixc(skyHor, skyTop, t);
            float gs = fmaxf(0.0f, dot3(d, sunDir)), gm = fmaxf(0.0f, dot3(d, moonDir));
            float sg = (powf(gs, 6.0f) * 0.22f + powf(gs, 80.0f) * 0.55f) * sunVis * dayAmount;
            float mg = powf(gm, 40.0f) * 0.30f * (1.0f - dayAmount) * (1.0f - rainAmt);
            glColor3f(c.r + sg * 0.9f + mg * 0.35f + fl * 0.5f, c.g + sg * 0.40f + mg * 0.45f + fl * 0.5f, c.b - sg * 0.30f + mg * 0.8f + fl * 0.6f);
            glVertex3f(d.x * R, d.y * R, d.z * R);
        }
        glEnd();
    }
    // stars
    float starA = (1.0f - dayAmount) * clampf(1.15f - cloudCover, 0.0f, 1.0f);
    if (starA > 0.02f) {
        glEnable(GL_POINT_SMOOTH); glPointSize(1.8f); glColor4f(1, 1, 1, starA);
        glBegin(GL_POINTS);
        for (size_t i = 0; i + 2 < stars.size(); i += 3) glVertex3f(stars[i], stars[i + 1], stars[i + 2]);
        glEnd();
    }
    glPopMatrix();

    // sun and moon billboards
    beginBillboards();
    glEnable(GL_TEXTURE_2D); glEnable(GL_BLEND);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
    V3 sp = camPos + sunDir * 450.0f, mp = camPos + moonDir * 450.0f;
    if (sunDir.y > -0.12f && sunVis > 0.02f) {
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);                    // normal blending keeps the colours orange, not white
        glBindTexture(GL_TEXTURE_2D, texGlow);
        glColor4f(1.0f, 0.55f, 0.12f, 0.62f * sunVis); sprite(sp, 210.0f);      // outer glow: orange
        glColor4f(1.0f, 0.74f, 0.20f, 0.85f * sunVis); sprite(sp, 78.0f);       // inner glow: golden yellow
        glBindTexture(GL_TEXTURE_2D, texSun);
        glColor4f(1.0f, 0.68f, 0.08f, sunVis); sprite(sp, 17.0f);               // sun disc: orange
        glColor4f(1.0f, 0.90f, 0.30f, sunVis); sprite(sp, 11.0f);               // hot core: yellow
    }
    if (moonDir.y > -0.12f && (1.0f - dayAmount) > 0.02f) {
        float mv = (1.0f - dayAmount) * (1.0f - 0.9f * rainAmt);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE);
        glBindTexture(GL_TEXTURE_2D, texGlow);
        glColor4f(0.55f, 0.66f, 1.0f, 0.42f * mv); sprite(mp, 95.0f);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glBindTexture(GL_TEXTURE_2D, texMoon);
        glColor4f(1, 1, 1, mv); sprite(mp, 15.0f);
    }

    // clouds (soft textured puffs)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glBindTexture(GL_TEXTURE_2D, texCloud);
    setFog();
    Col dayT = mixc(Col{1.0f, 1.0f, 1.0f}, Col{0.30f, 0.32f, 0.36f}, cloudDark);
    float nb = 0.11f + 0.89f * dayAmount;
    Col t = {dayT.r * nb + 0.02f, dayT.g * nb + 0.03f, dayT.b * nb + 0.06f};
    float fl2 = flash * 0.8f;
    for (size_t i = 0; i < puffs.size(); ++i) {
        const Puff &p = puffs[i];
        float a = clampf((cloudCover - p.thr) * 6.0f, 0.0f, 1.0f) * 0.92f;
        if (a < 0.01f) continue;
        V3 pd = norm3(V3{p.x - camPos.x, p.y - camPos.y, p.z - camPos.z});
        a *= 1.0f - 0.9f * smoothstep(0.90f, 0.975f, dot3(pd, sunDir)) * sunVis * dayAmount;
        glColor4f(clampf(t.r * p.tone + fl2, 0, 1), clampf(t.g * p.tone + fl2, 0, 1), clampf(t.b * p.tone + fl2, 0, 1), a);
        sprite(V3{p.x, p.y, p.z}, p.size);
    }
    glDisable(GL_FOG); glDisable(GL_TEXTURE_2D); glDisable(GL_BLEND);
    glDepthMask(GL_TRUE); glEnable(GL_DEPTH_TEST);
}

// ----------------------------------------------------------------------------
//  Terrain, trees, cabin ...
// ----------------------------------------------------------------------------
static void drawTerrain() {
    glEnableClientState(GL_VERTEX_ARRAY); glEnableClientState(GL_NORMAL_ARRAY); glEnableClientState(GL_COLOR_ARRAY);
    glVertexPointer(3, GL_FLOAT, 0, &vPos[0]); glNormalPointer(GL_FLOAT, 0, &vNrm[0]); glColorPointer(3, GL_FLOAT, 0, &vCol[0]);
    glDrawElements(GL_TRIANGLES, (GLsizei)vIdx.size(), GL_UNSIGNED_INT, &vIdx[0]);
    glDisableClientState(GL_VERTEX_ARRAY); glDisableClientState(GL_NORMAL_ARRAY); glDisableClientState(GL_COLOR_ARRAY);

    // flat skirt around the terrain so the horizon never shows an edge
    Col g = mixc(Col{0.25f, 0.45f, 0.13f}, Col{0.93f, 0.95f, 0.99f}, snowCover);
    glColor3f(g.r, g.g, g.b); glNormal3f(0, 1, 0);
    const float x0 = TX0 + (TNX - 1) * TCELL, z0 = TZ0 + (TNZ - 1) * TCELL, B = 3000.0f, y = 0.98f;
    glBegin(GL_QUADS);
    glVertex3f(-B, y, -B); glVertex3f(TX0, y, -B); glVertex3f(TX0, y, B); glVertex3f(-B, y, B);
    glVertex3f(x0, y, -B); glVertex3f(B, y, -B); glVertex3f(B, y, B); glVertex3f(x0, y, B);
    glVertex3f(TX0, y, -B); glVertex3f(x0, y, -B); glVertex3f(x0, y, TZ0); glVertex3f(TX0, y, TZ0);
    glVertex3f(TX0, y, z0); glVertex3f(x0, y, z0); glVertex3f(x0, y, B); glVertex3f(TX0, y, B);
    glEnd();
}

static void cullTrees() {
    visTrees.clear();
    float th = tanf(rad(fovY * 0.5f)) * ((float)winW / winH) * 1.15f;
    for (size_t i = 0; i < trees.size(); ++i) {
        const Tree &t = trees[i];
        V3 d = {t.x - camPos.x, t.y - camPos.y, t.z - camPos.z};
        float f = dot3(d, camFwd), s = dot3(d, camRight);
        if (f < -t.w - 6.0f || f > 520.0f) continue;
        if (fabsf(s) > f * th + t.w + 10.0f) continue;
        visTrees.push_back((int)i);
    }
}

static void drawTrees(bool mirror) {
    Col snowC = {0.90f, 0.93f, 0.98f};
    float sway = (0.6f + 2.2f * chop) ;
    for (size_t n = 0; n < visTrees.size(); ++n) {
        const Tree &t = trees[visTrees[n]];
        float dx = t.x - camPos.x, dz = t.z - camPos.z, d2 = dx * dx + dz * dz;
        bool farLod = mirror || d2 > 150.0f * 150.0f;
        glPushMatrix();
        glTranslatef(t.x, t.y, t.z);
        if (!farLod) glRotatef(sinf(timeSec * 1.1f + t.x * 0.37f) * sway, 0, 0, 1);
        if (t.type == 0) {
            glScalef(t.w, t.h, t.w);
            Col f1 = {0.09f * t.v, 0.36f * t.v, 0.12f * t.v}, f2 = {0.13f * t.v, 0.44f * t.v, 0.14f * t.v};
            if (farLod) {
                Col c = mixc(f1, snowC, snowCover * 0.7f); glColor3f(c.r, c.g, c.b); glCallList(dlPineFar);
            } else {
                glColor3f(0.27f, 0.18f, 0.10f); glCallList(dlTrunk);
                Col c1 = mixc(f1, snowC, snowCover * 0.80f), c2 = mixc(f2, snowC, snowCover * 0.65f);
                glColor3f(c1.r, c1.g, c1.b); glCallList(dlPineA);
                glColor3f(c2.r, c2.g, c2.b); glCallList(dlPineB);
            }
        } else {
            glScalef(t.w, t.h, t.w);
            glColor3f(0.30f, 0.20f, 0.11f);
            glPushMatrix(); glScalef(0.09f, 0.55f, 0.09f); glTranslatef(0, 0.5f, 0); glScalef(1, 1, 1); glCallList(dlBall); glPopMatrix();
            Col leaf = mixc(Col{0.18f * t.v, 0.45f * t.v, 0.10f * t.v}, Col{0.55f * t.v, 0.45f * t.v, 0.10f}, (t.v - 0.75f) * 0.9f * (t.x > 0 ? 1.0f : 0.2f));
            Col lc = mixc(leaf, snowC, snowCover * 0.7f); glColor3f(lc.r, lc.g, lc.b);
            const float cr[3][4] = {{0, 0.68f, 0, 0.36f}, {0.20f, 0.56f, 0.10f, 0.26f}, {-0.17f, 0.78f, -0.08f, 0.27f}};
            for (int k = 0; k < 3; ++k) {
                glPushMatrix(); glTranslatef(cr[k][0], cr[k][1], cr[k][2]); glScalef(cr[k][3], cr[k][3] * 0.85f, cr[k][3]); glCallList(dlBall); glPopMatrix();
            }
        }
        glPopMatrix();
    }
}

static void box(float x, float y, float z, float sx, float sy, float sz) {
    glPushMatrix(); glTranslatef(x, y, z); glScalef(sx, sy, sz); glutSolidCube(1.0); glPopMatrix();
}
static void slab(float x, float y, float z, float angX, float sx, float sy, float sz) {
    glPushMatrix(); glTranslatef(x, y, z); glRotatef(angX, 1, 0, 0); glScalef(sx, sy, sz); glutSolidCube(1.0); glPopMatrix();
}

static void drawCabin() {
    glPushMatrix();
    glTranslatef(CABIN_X, CABIN_H, CABIN_Z);
    glRotatef(CABIN_YAW, 0, 1, 0);
    glColor3f(0.40f, 0.39f, 0.37f); box(0, 0.35f, 0, 9.2f, 1.0f, 7.2f);                        // stone base
    for (int i = 0; i < 8; ++i) {                                                               // log walls
        float y = 0.9f + 0.38f * i + 0.19f, s = (i & 1) ? 0.90f : 1.0f;
        glColor3f(0.52f * s, 0.32f * s, 0.16f * s); box(0, y, 0, 8.0f, 0.37f, 6.0f);
        glColor3f(0.44f * s, 0.26f * s, 0.12f * s);
        box(-4.1f, y, -3.1f, 0.55f, 0.35f, 0.55f); box(4.1f, y, -3.1f, 0.55f, 0.35f, 0.55f);
        box(-4.1f, y, 3.1f, 0.55f, 0.35f, 0.55f);  box(4.1f, y, 3.1f, 0.55f, 0.35f, 0.55f);
    }
    glColor3f(0.48f, 0.29f, 0.15f);                                                             // gable ends
    glBegin(GL_TRIANGLES);
    glNormal3f(1, 0, 0);  glVertex3f(4.0f, 3.94f, -3.0f); glVertex3f(4.0f, 3.94f, 3.0f); glVertex3f(4.0f, 6.1f, 0);
    glNormal3f(-1, 0, 0); glVertex3f(-4.0f, 3.94f, 3.0f); glVertex3f(-4.0f, 3.94f, -3.0f); glVertex3f(-4.0f, 6.1f, 0);
    glEnd();
    glColor3f(0.36f, 0.22f, 0.14f);                                                             // roof
    slab(0, 5.04f, 1.7f, 33.0f, 9.6f, 0.26f, 4.05f);
    slab(0, 5.04f, -1.7f, -33.0f, 9.6f, 0.26f, 4.05f);
    glColor3f(0.43f, 0.43f, 0.45f); box(2.6f, 5.4f, -1.7f, 1.0f, 3.3f, 1.0f); box(2.6f, 7.1f, -1.7f, 1.25f, 0.22f, 1.25f);  // chimney
    glColor3f(0.55f, 0.38f, 0.22f); box(0, 0.98f, 4.3f, 6.8f, 0.2f, 2.4f);                       // porch
    glColor3f(0.40f, 0.26f, 0.15f);
    box(-3.1f, 2.1f, 5.3f, 0.25f, 2.3f, 0.25f); box(3.1f, 2.1f, 5.3f, 0.25f, 2.3f, 0.25f);
    slab(0, 3.4f, 4.4f, 12.0f, 7.4f, 0.16f, 2.9f);
    glColor3f(0.27f, 0.16f, 0.09f); box(-1.7f, 2.0f, 3.03f, 1.3f, 2.1f, 0.12f);                 // door
    glColor3f(0.30f, 0.18f, 0.10f);
    box(1.9f, 2.4f, 3.03f, 1.9f, 1.5f, 0.14f); box(-3.0f, 2.4f, 3.03f, 1.0f, 1.3f, 0.14f);      // window frames

    if (snowCover > 0.03f) {                                                                    // snow on the roof
        float t = 0.10f + 0.55f * snowCover;
        glColor3f(0.95f, 0.97f, 1.0f);
        slab(0, 5.04f + 0.14f + t * 0.5f, 1.7f, 33.0f, 9.8f, t, 4.1f);
        slab(0, 5.04f + 0.14f + t * 0.5f, -1.7f, -33.0f, 9.8f, t, 4.1f);
        box(2.6f, 7.28f + t * 0.5f, -1.7f, 1.2f, t, 1.2f);
        slab(0, 3.5f + t * 0.5f, 4.4f, 12.0f, 7.5f, t * 0.8f, 3.0f);
        box(0, 1.1f + t * 0.3f, 4.3f, 6.8f, t * 0.6f, 2.4f);
    }
    // windows: glass by day, warm light at night
    glDisable(GL_LIGHTING);
    Col wc = mixc(Col{1.0f, 0.78f, 0.38f}, Col{0.50f, 0.64f, 0.74f}, dayAmount);
    glColor3f(wc.r, wc.g, wc.b);
    glBegin(GL_QUADS);
    glVertex3f(1.0f, 1.8f, 3.11f); glVertex3f(2.8f, 1.8f, 3.11f); glVertex3f(2.8f, 3.0f, 3.11f); glVertex3f(1.0f, 3.0f, 3.11f);
    glVertex3f(-3.4f, 1.9f, 3.11f); glVertex3f(-2.6f, 1.9f, 3.11f); glVertex3f(-2.6f, 2.9f, 3.11f); glVertex3f(-3.4f, 2.9f, 3.11f);
    glEnd();
    glColor3f(0.30f, 0.18f, 0.10f);
    glBegin(GL_QUADS);
    glVertex3f(1.87f, 1.8f, 3.12f); glVertex3f(1.93f, 1.8f, 3.12f); glVertex3f(1.93f, 3.0f, 3.12f); glVertex3f(1.87f, 3.0f, 3.12f);
    glVertex3f(1.0f, 2.37f, 3.12f); glVertex3f(2.8f, 2.37f, 3.12f); glVertex3f(2.8f, 2.43f, 3.12f); glVertex3f(1.0f, 2.43f, 3.12f);
    glEnd();
    glEnable(GL_LIGHTING);
    glPopMatrix();
}

static void drawCabinGlow() {
    float k = 1.0f - dayAmount;
    if (k < 0.03f) return;
    glDisable(GL_LIGHTING); glDisable(GL_FOG); glDepthMask(GL_FALSE);
    glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE); glEnable(GL_TEXTURE_2D); glBindTexture(GL_TEXTURE_2D, texGlow);
    beginBillboards();
    float yw = rad(CABIN_YAW), c = cosf(yw), s = sinf(yw);
    // local (lx, ly, lz) -> world
    const float L[2][3] = {{1.9f, 2.4f, 3.3f}, {-3.0f, 2.4f, 3.3f}};
    for (int i = 0; i < 2; ++i) {
        V3 p = {CABIN_X + L[i][0] * c + L[i][2] * s, CABIN_H + L[i][1], CABIN_Z - L[i][0] * s + L[i][2] * c};
        glColor4f(1.0f, 0.72f, 0.35f, 0.65f * k); sprite(p, i == 0 ? 5.5f : 3.5f);
    }
    glDisable(GL_TEXTURE_2D); glDisable(GL_BLEND); glDepthMask(GL_TRUE); glEnable(GL_LIGHTING);
}

static void drawFence() {
    Col wood = mixc(Col{0.42f, 0.29f, 0.17f}, Col{0.55f, 0.55f, 0.58f}, 0.0f);
    glColor3f(wood.r, wood.g, wood.b);
    for (size_t i = 0; i < fencePosts.size(); ++i) {
        const V3 &p = fencePosts[i];
        box(p.x, p.y + 0.55f, p.z, 0.16f, 1.1f, 0.16f);
        if (snowCover > 0.05f) { glColor3f(0.95f, 0.97f, 1.0f); box(p.x, p.y + 1.1f + 0.05f * snowCover, p.z, 0.22f, 0.12f * snowCover + 0.02f, 0.22f); glColor3f(wood.r, wood.g, wood.b); }
        if (i + 1 < fencePosts.size()) {
            const V3 &q = fencePosts[i + 1];
            for (int r = 0; r < 2; ++r) {
                float hh = r == 0 ? 0.42f : 0.85f;
                V3 a = {p.x, p.y + hh, p.z}, b = {q.x, q.y + hh, q.z};
                V3 d = {b.x - a.x, b.y - a.y, b.z - a.z};
                float len = sqrtf(dot3(d, d)), yaw = atan2f(-d.z, d.x) * 180.0f / PI, pit = asinf(d.y / len) * 180.0f / PI;
                glPushMatrix(); glTranslatef((a.x + b.x) * 0.5f, (a.y + b.y) * 0.5f, (a.z + b.z) * 0.5f);
                glRotatef(yaw, 0, 1, 0); glRotatef(pit, 0, 0, 1); glScalef(len, 0.1f, 0.07f); glutSolidCube(1.0); glPopMatrix();
            }
        }
    }
}

static void drawRocks() {
    for (size_t i = 0; i < rocks.size(); ++i) {
        const Rock &r = rocks[i];
        Col c = mixc(Col{0.42f * r.shade, 0.41f * r.shade, 0.40f * r.shade}, Col{0.93f, 0.95f, 0.99f}, snowCover * 0.8f);
        glColor3f(c.r, c.g, c.b);
        glPushMatrix(); glTranslatef(r.x, r.y + r.sy * 0.25f, r.z); glRotatef(r.rot, 0, 1, 0); glScalef(r.sx, r.sy, r.sz);
        glCallList(dlBall); glPopMatrix();
    }
}

static void drawGrass() {
    float k = 1.0f - 0.55f * snowCover;
    Col snowC = {0.90f, 0.93f, 0.98f};
    glNormal3f(0, 1, 0);
    glBegin(GL_TRIANGLES);
    for (size_t i = 0; i < tufts.size(); ++i) {
        const Tuft &t = tufts[i];
        float dx = t.x - camPos.x, dz = t.z - camPos.z;
        if (dx * dx + dz * dz > 95.0f * 95.0f) continue;
        if (dx * camFwd.x + dz * camFwd.z < -3.0f) continue;
        Col c = mixc(t.c, snowC, snowCover * 0.75f), b = scalec(c, 0.55f);
        float h = t.h * k;
        for (int b2 = 0; b2 < 3; ++b2) {
            float a = t.rot + b2 * 2.09f, ca = cosf(a), sa = sinf(a), lean = 0.18f * h;
            glColor3f(b.r, b.g, b.b); glVertex3f(t.x - ca * 0.06f, t.y, t.z - sa * 0.06f); glVertex3f(t.x + ca * 0.06f, t.y, t.z + sa * 0.06f);
            glColor3f(c.r, c.g, c.b); glVertex3f(t.x + sa * lean, t.y + h, t.z - ca * lean);
        }
    }
    glEnd();
    float fa = (1.0f - snowCover) * (1.0f - 0.5f * rainAmt);
    if (fa > 0.05f) {
        glDisable(GL_LIGHTING); glPointSize(4.0f); glEnable(GL_POINT_SMOOTH);
        float lum = 0.25f + 0.75f * dayAmount * (0.6f + 0.4f * dimLight);
        glBegin(GL_POINTS);
        for (size_t i = 0; i < flowers.size(); ++i) {
            const Flower &f = flowers[i];
            float dx = f.x - camPos.x, dz = f.z - camPos.z;
            if (dx * dx + dz * dz > 80.0f * 80.0f || dx * camFwd.x + dz * camFwd.z < 0) continue;
            glColor3f(f.c.r * lum, f.c.g * lum, f.c.b * lum); glVertex3f(f.x, f.y, f.z);
        }
        glEnd();
        glEnable(GL_LIGHTING);
    }
}

// ----------------------------------------------------------------------------
//  Water (with real reflection through the stencil buffer)
// ----------------------------------------------------------------------------
static const float WX0 = -58.0f, WX1 = 96.0f, WZ0 = -68.0f, WZ1 = 34.0f, WCELL = 1.5f;

static void drawWaterPlane() {
    glBegin(GL_QUADS);
    glVertex3f(WX0, 0, WZ0); glVertex3f(WX0, 0, WZ1); glVertex3f(WX1, 0, WZ1); glVertex3f(WX1, 0, WZ0);
    glEnd();
}

static void waveSlope(float x, float z, float t, float amp, float &sx, float &sz) {
    static const float W[5][4] = {{0.8f, 0.6f, 0.45f, 1.3f}, {-0.5f, 0.85f, 0.8f, 1.7f}, {0.3f, -0.95f, 1.3f, 2.1f}, {0.95f, 0.3f, 2.2f, 2.9f}, {-0.7f, -0.7f, 3.4f, 3.7f}};
    static const float A[5] = {1.0f, 0.75f, 0.5f, 0.32f, 0.2f};
    sx = sz = 0;
    for (int i = 0; i < 5; ++i) {
        float c = cosf(W[i][2] * (W[i][0] * x + W[i][1] * z) + W[i][3] * t) * A[i] * W[i][2] * amp;
        sx += c * W[i][0]; sz += c * W[i][1];
    }
}

static void drawWater() {
    setupLights(); setFog();
    glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    float spec[4] = {0.95f, 0.95f, 0.88f, 1}, none[4] = {0, 0, 0, 1};
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, spec); glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 110.0f);

    Col deep = mixc(Col{0.03f, 0.22f, 0.30f}, Col{0.05f, 0.17f, 0.24f}, snowAmt);
    deep = mixc(deep, scalec(skyHor, 0.45f), 0.35f * cloudDark + 0.1f);
    deep = mixc(Col{deep.r * 0.55f, deep.g * 0.7f, deep.b * 1.0f}, deep, 0.2f + 0.8f * dayAmount);
    float amp = 0.10f + 0.20f * chop;
    int nx = (int)((WX1 - WX0) / WCELL), nz = (int)((WZ1 - WZ0) / WCELL);
    for (int j = 0; j < nz; ++j) {
        glBegin(GL_TRIANGLE_STRIP);
        for (int i = 0; i <= nx; ++i) for (int s = 0; s < 2; ++s) {
            float x = WX0 + i * WCELL, z = WZ0 + (j + s) * WCELL, sx, sz;
            waveSlope(x, z, timeSec, amp, sx, sz);
            V3 n = norm3(V3{-sx, 1.0f, -sz});
            V3 v = norm3(V3{camPos.x - x, camPos.y, camPos.z - z});
            float cs = clampf(dot3(n, v), 0, 1);
            float a = clampf(0.80f - 0.58f * powf(1.0f - cs, 2.0f), 0.22f, 0.85f);
            a = lerpf(a, fmaxf(a, 0.78f), rainAmt * 0.6f);
            glColor4f(deep.r, deep.g, deep.b, a); glNormal3f(n.x, n.y, n.z); glVertex3f(x, 0, z);
        }
        glEnd();
    }
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, none);
    glDisable(GL_BLEND);
}


// ----------------------------------------------------------------------------
//  Children playing: two kick a ball, three play tag, one jumps and waves
// ----------------------------------------------------------------------------
struct Kid { int role; float cx, cz, r, ph, w; Col shirt, pants, hair, skin; };
static vector<Kid> kids;
static const float KA_X = -22.0f, KA_Z = 24.0f, KB_X = -10.0f, KB_Z = 27.0f;   // the two ball players

static void initKids() {
    const Col shirts[6] = {{0.90f, 0.20f, 0.20f}, {0.95f, 0.75f, 0.15f}, {0.20f, 0.50f, 0.90f}, {0.95f, 0.45f, 0.70f}, {0.20f, 0.70f, 0.35f}, {0.95f, 0.55f, 0.15f}};
    const Col pants[3]  = {{0.20f, 0.25f, 0.50f}, {0.40f, 0.30f, 0.20f}, {0.25f, 0.25f, 0.30f}};
    const Col hairs[3]  = {{0.08f, 0.06f, 0.05f}, {0.35f, 0.20f, 0.08f}, {0.12f, 0.10f, 0.10f}};
    const Col skins[3]  = {{0.95f, 0.78f, 0.62f}, {0.80f, 0.60f, 0.45f}, {0.62f, 0.44f, 0.32f}};
    for (int i = 0; i < 6; ++i) {
        Kid k; k.role = 0; k.cx = k.cz = k.r = k.ph = k.w = 0;
        k.shirt = shirts[i]; k.pants = pants[i % 3]; k.hair = hairs[(i + 1) % 3]; k.skin = skins[(i * 2) % 3];
        kids.push_back(k);
    }
    for (int i = 0; i < 3; ++i) { kids[i].role = 0; kids[i].cx = 1.0f; kids[i].cz = 25.0f; kids[i].r = 5.0f; kids[i].ph = i * 0.9f; kids[i].w = 0.62f; }
    kids[3].role = 1; kids[3].ph = 0.0f;                                // ball player A
    kids[4].role = 1; kids[4].ph = 1.0f;                                // ball player B
    kids[5].role = 2; kids[5].cx = -31.0f; kids[5].cz = 27.0f;          // jumping child
}

static void blob(float x, float z, float r, float a) {                  // soft contact shadow on the ground
    glDisable(GL_LIGHTING); glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); glDepthMask(GL_FALSE);
    glColor4f(0, 0, 0, a);
    glBegin(GL_TRIANGLE_FAN);
    glVertex3f(x, heightAt(x, z) + 0.06f, z);
    for (int i = 0; i <= 14; ++i) {
        float an = 2 * PI * i / 14, vx = x + cosf(an) * r, vz = z + sinf(an) * r;
        glVertex3f(vx, heightAt(vx, vz) + 0.06f, vz);
    }
    glEnd();
    glDepthMask(GL_TRUE); glDisable(GL_BLEND); glEnable(GL_LIGHTING);
}

static void drawKid(const Kid &k) {
    const float t = timeSec;
    float x = k.cx, z = k.cz, yaw = 0, hop = 0, legL = 0, legR = 0, armL = 0, armR = 0;
    if (k.role == 0) {                                                  // running in circles (tag)
        float a = k.ph + k.w * t, vx = -k.r * sinf(a), vz = k.r * 0.7f * cosf(a);
        x = k.cx + k.r * cosf(a); z = k.cz + k.r * 0.7f * sinf(a);
        yaw = atan2f(vx, vz) * 180.0f / PI;
        float s = sinf(t * 10.0f + k.ph * 3.0f);
        legL = 50.0f * s; legR = -50.0f * s; armL = -45.0f * s; armR = 45.0f * s; hop = fabsf(s) * 0.08f;
    } else if (k.role == 1) {                                           // kicking the ball to each other
        bool isA = k.ph < 0.5f;
        x = isA ? KA_X : KB_X; z = isA ? KA_Z : KB_Z;
        float ox = isA ? KB_X : KA_X, oz = isA ? KB_Z : KA_Z;
        yaw = atan2f(ox - x, oz - z) * 180.0f / PI;
        float cyc = t / 1.9f; int n = (int)floorf(cyc); float f = cyc - n;
        bool launching = isA ? (n % 2 == 0) : (n % 2 == 1);
        if (launching && f < 0.22f) legR = -70.0f * sinf(f / 0.22f * PI);
        armL = 25.0f * sinf(t * 3.0f + k.ph); armR = -armL; hop = 0.03f * fabsf(sinf(t * 4.0f));
    } else {                                                            // jumping and waving at the viewer
        yaw = atan2f(camPos.x - x, camPos.z - z) * 180.0f / PI;
        hop = fabsf(sinf(t * 6.0f)) * 0.45f;
        armL = -150.0f - 25.0f * sinf(t * 12.0f); armR = -150.0f + 25.0f * sinf(t * 12.0f);
        legL = legR = -hop * 40.0f;
    }
    const float S = 1.35f * kidVis, y = heightAt(x, z);
    blob(x, z, 0.5f * S * (1.0f - 0.4f * hop), 0.32f * kidVis);

    glPushMatrix();
    glTranslatef(x, y + hop * S, z); glRotatef(yaw, 0, 1, 0); glScalef(S, S, S);
    for (int side = -1; side <= 1; side += 2) {                          // legs and shoes
        glPushMatrix(); glTranslatef(side * 0.09f, 0.50f, 0); glRotatef(side < 0 ? legL : legR, 1, 0, 0);
        glColor3f(k.pants.r, k.pants.g, k.pants.b);
        glPushMatrix(); glTranslatef(0, -0.24f, 0); glScalef(0.12f, 0.48f, 0.14f); glutSolidCube(1.0); glPopMatrix();
        glColor3f(0.15f, 0.12f, 0.12f);
        glPushMatrix(); glTranslatef(0, -0.50f, 0.03f); glScalef(0.13f, 0.07f, 0.22f); glutSolidCube(1.0); glPopMatrix();
        glPopMatrix();
    }
    glColor3f(k.shirt.r, k.shirt.g, k.shirt.b);                           // body
    glPushMatrix(); glTranslatef(0, 0.70f, 0); glScalef(0.30f, 0.40f, 0.18f); glutSolidCube(1.0); glPopMatrix();
    for (int side = -1; side <= 1; side += 2) {                          // arms and hands
        glPushMatrix(); glTranslatef(side * 0.20f, 0.88f, 0); glRotatef(side < 0 ? armL : armR, 1, 0, 0);
        glColor3f(k.shirt.r, k.shirt.g, k.shirt.b);
        glPushMatrix(); glTranslatef(0, -0.19f, 0); glScalef(0.09f, 0.38f, 0.09f); glutSolidCube(1.0); glPopMatrix();
        glColor3f(k.skin.r, k.skin.g, k.skin.b);
        glPushMatrix(); glTranslatef(0, -0.41f, 0); glScalef(0.055f, 0.055f, 0.055f); glCallList(dlBall); glPopMatrix();
        glPopMatrix();
    }
    glColor3f(k.skin.r, k.skin.g, k.skin.b);                              // head
    glPushMatrix(); glTranslatef(0, 1.05f, 0); glScalef(0.15f, 0.16f, 0.15f); glCallList(dlBall); glPopMatrix();
    glColor3f(k.hair.r, k.hair.g, k.hair.b);                              // hair
    glPushMatrix(); glTranslatef(0, 1.10f, -0.02f); glScalef(0.16f, 0.12f, 0.16f); glCallList(dlBall); glPopMatrix();
    glColor3f(0.05f, 0.05f, 0.05f);                                       // eyes
    for (int side = -1; side <= 1; side += 2) {
        glPushMatrix(); glTranslatef(side * 0.055f, 1.07f, 0.135f); glScalef(0.018f, 0.018f, 0.018f); glCallList(dlBall); glPopMatrix();
    }
    glPopMatrix();
}

static void drawBall() {
    const float dx = KB_X - KA_X, dz = KB_Z - KA_Z, len = sqrtf(dx * dx + dz * dz), ux = dx / len, uz = dz / len;
    float cyc = timeSec / 1.9f; int n = (int)floorf(cyc); float f = cyc - n;
    bool aToB = (n % 2 == 0);
    float sx = (aToB ? KA_X : KB_X) + (aToB ? ux : -ux) * 0.6f, sz = (aToB ? KA_Z : KB_Z) + (aToB ? uz : -uz) * 0.6f;
    float ex = (aToB ? KB_X : KA_X) - (aToB ? ux : -ux) * 0.6f, ez = (aToB ? KB_Z : KA_Z) - (aToB ? uz : -uz) * 0.6f;
    float p = clampf((f - 0.12f) / 0.88f, 0.0f, 1.0f);
    float x = lerpf(sx, ex, p), z = lerpf(sz, ez, p), h = 0.25f + 2.0f * 4.0f * p * (1.0f - p);
    blob(x, z, 0.28f * kidVis, 0.30f * kidVis / (1.0f + 0.5f * h));
    glColor3f(0.95f, 0.25f, 0.20f);
    glPushMatrix(); glTranslatef(x, heightAt(x, z) + h, z); glScalef(0.22f * kidVis, 0.22f * kidVis, 0.22f * kidVis); glCallList(dlBall); glPopMatrix();
}

static void drawKids() {
    if (kidVis < 0.03f) return;
    for (size_t i = 0; i < kids.size(); ++i) drawKid(kids[i]);
    drawBall();
}

static void drawWorld(bool mirror) {
    drawSky();
    setFog();
    setupLights();
    drawTerrain();
    drawTrees(mirror);
    drawCabin();
    drawFence();
    drawRocks();
    if (!mirror) drawGrass();
    if (!mirror) drawKids();
    if (!mirror) drawCabinGlow();
}

static void drawReflection() {
    glEnable(GL_STENCIL_TEST);
    glStencilFunc(GL_ALWAYS, 1, 1); glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
    glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE); glDepthMask(GL_FALSE);
    glDisable(GL_LIGHTING); glDisable(GL_FOG);
    drawWaterPlane();                                                  // mark visible lake pixels
    glStencilFunc(GL_EQUAL, 1, 1); glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
    glDepthMask(GL_TRUE); glDepthFunc(GL_ALWAYS); glDepthRange(1.0, 1.0);
    drawWaterPlane();                                                  // clear depth there
    glDepthRange(0.0, 1.0); glDepthFunc(GL_LESS);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

    glPushMatrix();
    glScalef(1.0f, -1.0f, 1.0f);                                       // mirror the world in the water plane
    double eq[4] = {0, 1, 0, 0};
    glClipPlane(GL_CLIP_PLANE0, eq); glEnable(GL_CLIP_PLANE0);
    drawWorld(true);
    glDisable(GL_CLIP_PLANE0);
    glPopMatrix();
    glDisable(GL_STENCIL_TEST);
}

// ----------------------------------------------------------------------------
//  Waterfall, ripples, birds, weather particles
// ----------------------------------------------------------------------------
static void drawWaterfall() {
    if (fallPath.size() < 3) return;
    glDisable(GL_LIGHTING); glDepthMask(GL_FALSE); glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    setFog();
    glEnable(GL_POLYGON_OFFSET_FILL); glPolygonOffset(-2.0f, -2.0f);
    float lum = 0.35f + 0.65f * dayAmount;
    glBegin(GL_QUAD_STRIP);
    for (size_t i = 0; i < fallPath.size(); ++i) {
        const V3 &p = fallPath[i];
        float w = 1.5f + 0.02f * (float)i;
        V3 t = (i + 1 < fallPath.size()) ? norm3(V3{fallPath[i + 1].x - p.x, 0, fallPath[i + 1].z - p.z}) : V3{0, 0, 1};
        float nx = -t.z, nz = t.x;
        float a = clampf(0.78f + 0.22f * sinf((float)i * 0.9f - timeSec * 7.0f + vnoise((float)i * 0.4f, 3.0f) * 6.0f), 0.35f, 1.0f);
        if (i < 3) a *= (i + 1) / 4.0f;
        glColor4f(0.92f * lum, 0.96f * lum, 1.0f * lum, a);
        glVertex3f(p.x - nx * w, p.y + 0.35f, p.z - nz * w); glVertex3f(p.x + nx * w, p.y + 0.35f, p.z + nz * w);
    }
    glEnd();
    glDisable(GL_POLYGON_OFFSET_FILL);
    // mist at the bottom
    const V3 &e = fallPath.back();
    glEnable(GL_TEXTURE_2D); glBindTexture(GL_TEXTURE_2D, texGlow); beginBillboards();
    glColor4f(0.9f * lum, 0.95f * lum, 1.0f * lum, 0.35f + 0.1f * sinf(timeSec * 2.0f)); sprite(V3{e.x, e.y + 2.0f, e.z}, 12.0f);
    glDisable(GL_TEXTURE_2D); glDisable(GL_FOG); glDisable(GL_BLEND); glDepthMask(GL_TRUE);
}

static void drawRipples() {
    if (rainAmt < 0.1f) return;
    glDisable(GL_LIGHTING); glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); glDepthMask(GL_FALSE);
    glLineWidth(1.2f);
    for (size_t i = 0; i < ripples.size(); ++i) {
        const Ripple &r = ripples[i];
        if (r.age > r.life) continue;
        float t = r.age / r.life, ringR = 0.2f + 2.0f * t;
        glColor4f(0.8f, 0.9f, 1.0f, (1.0f - t) * 0.6f * rainAmt);
        glBegin(GL_LINE_LOOP);
        for (int k = 0; k < 20; ++k) { float a = 2 * PI * k / 20; glVertex3f(r.x + cosf(a) * ringR, 0.03f, r.z + sinf(a) * ringR); }
        glEnd();
    }
    glDepthMask(GL_TRUE); glDisable(GL_BLEND);
}

static void drawBirds() {
    float v = sunVis * dayAmount * (1.0f - snowAmt);
    if (v < 0.05f) return;
    glDisable(GL_LIGHTING); glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    setFog();
    for (size_t i = 0; i < birds.size(); ++i) {
        const Bird &b = birds[i];
        float a = b.phase * 0.5f + (float)i, x = b.cx + cosf(a) * b.r, z = b.cz + sinf(a) * b.r, y = b.y + sinf(a * 2.0f) * 2.0f;
        float hd = -sinf(a) * 180.0f / PI;                             // face the direction of flight
        float flap = sinf(timeSec * b.flap + (float)i) * 0.7f;
        glPushMatrix(); glTranslatef(x, y, z); glRotatef(hd - 90.0f, 0, 1, 0);
        glColor4f(0.10f, 0.10f, 0.12f, 0.9f * v);
        glBegin(GL_TRIANGLES);
        glVertex3f(0, 0, 0.5f); glVertex3f(-1.1f, flap, -0.2f); glVertex3f(0, 0, -0.5f);
        glVertex3f(0, 0, 0.5f); glVertex3f(1.1f, flap, -0.2f); glVertex3f(0, 0, -0.5f);
        glEnd();
        glPopMatrix();
    }
    glDisable(GL_FOG); glDisable(GL_BLEND);
}

static void drawParticles() {
    glDisable(GL_LIGHTING); glDisable(GL_FOG); glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); glDepthMask(GL_FALSE);
    float lum = 0.35f + 0.65f * dayAmount + flash * 0.4f;
    if (rainAmt > 0.02f) {
        glLineWidth(1.0f); glEnable(GL_LINE_SMOOTH);
        glColor4f(0.78f * lum, 0.85f * lum, 1.0f * lum, 0.38f * rainAmt);
        glBegin(GL_LINES);
        for (size_t i = 0; i < rain.size(); ++i) {
            const Drop &p = rain[i];
            glVertex3f(p.x, p.y, p.z); glVertex3f(p.x, p.y + p.speed * 0.038f, p.z);   // straight vertical streak
        }
        glEnd();
    }
    if (snowAmt > 0.02f) {
        glEnable(GL_POINT_SMOOTH);
        float l2 = 0.55f + 0.45f * dayAmount;
        glColor4f(l2, l2, l2, 0.95f * snowAmt);
        const float sizes[3] = {2.2f, 3.4f, 4.8f};
        for (int pass = 0; pass < 3; ++pass) {
            glPointSize(sizes[pass]); glBegin(GL_POINTS);
            for (size_t i = pass; i < snow.size(); i += 3) glVertex3f(snow[i].x, snow[i].y, snow[i].z);
            glEnd();
        }
    }
    glDepthMask(GL_TRUE); glDisable(GL_BLEND);
}

// ----------------------------------------------------------------------------
//  HUD
// ----------------------------------------------------------------------------
static void text(float x, float y, const char *s, void *font = GLUT_BITMAP_HELVETICA_12) {
    glRasterPos2f(x, y); for (; *s; ++s) glutBitmapCharacter(font, *s);
}
static void panel(float x0, float y0, float x1, float y1) {
    glColor4f(0.04f, 0.07f, 0.12f, 0.72f);
    glBegin(GL_QUADS); glVertex2f(x0, y0); glVertex2f(x1, y0); glVertex2f(x1, y1); glVertex2f(x0, y1); glEnd();
}
static void drawHUD() {
    glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity(); gluOrtho2D(0, winW, 0, winH);
    glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity();
    glDisable(GL_DEPTH_TEST); glDisable(GL_LIGHTING); glDisable(GL_FOG); glDisable(GL_TEXTURE_2D);
    glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    const char *label = "Sunny Mode";
    if (weatherType == 0) label = isDay ? "Sunny Mode" : "Night Mode";
    else if (weatherType == 1) label = isDay ? "Rainy Mode" : "Rainy Night";
    else label = isDay ? "Snowy Mode" : "Snowy Night";
    panel(14, winH - 52, 14 + 30 + 11.0f * (float)strlen(label), winH - 14);
    glColor3f(1, 1, 1); text(30, winH - 40, label, GLUT_BITMAP_HELVETICA_18);

    if (showHelp) {
        float x0 = winW - 270.0f, y1 = winH - 14.0f;
        panel(x0, y1 - 232, winW - 14.0f, y1);
        glColor3f(1, 1, 1); text(x0 + 14, y1 - 24, "Keyboard Controls", GLUT_BITMAP_HELVETICA_18);
        const char *l[] = {"1 - Sunny Mode", "2 - Rainy Mode", "3 - Snowy Mode", "4 - Night Mode", "W / S - Move Forward / Backward",
                           "A / D - Move Left / Right", "Arrows - Look Up/Down, Turn", "Shift - Run     N - Day/Night", "P - Pause   R - Reset   F - Fullscreen",
                           "H - Hide help   Esc - Exit"};
        glColor3f(0.85f, 0.9f, 1.0f);
        for (int i = 0; i < 10; ++i) text(x0 + 14, y1 - 52 - i * 17, l[i]);
    }
    char fps[32]; snprintf(fps, sizeof(fps), "%.0f FPS", fpsShown);
    glColor3f(1, 1, 1); text(16, 14, fps);

    glDisable(GL_BLEND); glEnable(GL_DEPTH_TEST);
    glMatrixMode(GL_PROJECTION); glPopMatrix(); glMatrixMode(GL_MODELVIEW); glPopMatrix();
}

// ----------------------------------------------------------------------------
//  GLUT callbacks
// ----------------------------------------------------------------------------
static void display() {
    glClearColor(skyHor.r, skyHor.g, skyHor.b, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION); glLoadIdentity();
    gluPerspective(fovY, (double)winW / (double)winH, 0.3, 1500.0);
    glMatrixMode(GL_MODELVIEW); glLoadIdentity();
    float yw = rad(camYaw), pt = rad(camPitch);
    camFwd = V3{sinf(yw) * cosf(pt), sinf(pt), -cosf(yw) * cosf(pt)};
    camRight = V3{cosf(yw), 0, sinf(yw)};
    gluLookAt(camPos.x, camPos.y, camPos.z, camPos.x + camFwd.x, camPos.y + camFwd.y, camPos.z + camFwd.z, 0, 1, 0);

    cullTrees();
    drawWorld(false);
    drawReflection();
    drawWater();
    drawWaterfall();
    drawRipples();
    drawBirds();
    drawParticles();
    glDisable(GL_LIGHTING); glDisable(GL_FOG);
    drawHUD();
    glutSwapBuffers();
}

static void reshape(int w, int h) { winW = w; winH = h > 0 ? h : 1; glViewport(0, 0, winW, winH); }

static void timer(int) {
    static int last = glutGet(GLUT_ELAPSED_TIME);
    static float acc = 0; static int frames = 0;
    int now = glutGet(GLUT_ELAPSED_TIME);
    float dt = clampf((now - last) / 1000.0f, 0.0f, 0.1f);
    last = now;
    acc += dt; ++frames;
    if (acc > 0.5f) { fpsShown = frames / acc; acc = 0; frames = 0; }
    updateCamera(dt);
    update(dt);
    glutPostRedisplay();
    glutTimerFunc(8, timer, 0);
}

static void keyboard(unsigned char key, int, int) {
    shiftHeld = (glutGetModifiers() & GLUT_ACTIVE_SHIFT) != 0;
    int k = tolower(key); keyDown[k] = true;
    switch (k) {
        case '1': setMode(0, true); break;
        case '2': setMode(1, true); break;
        case '3': setMode(2, true); break;
        case '4': setMode(0, false); break;
        case 'n': isDay = !isDay; break;
        case 'p': paused = !paused; break;
        case 'h': showHelp = !showHelp; break;
        case 'r': camPos.x = 6; camPos.z = 54; camYaw = -16; camPitch = -2; fovY = 58; break;
        case 'f': fullscreen = !fullscreen; if (fullscreen) glutFullScreen(); else { glutReshapeWindow(1280, 720); glutPositionWindow(80, 60); } break;
        case 27: exit(0);
    }
}
static void keyboardUp(unsigned char key, int, int) { shiftHeld = (glutGetModifiers() & GLUT_ACTIVE_SHIFT) != 0; keyDown[tolower(key)] = false; }
static void special(int k, int, int) {
    shiftHeld = (glutGetModifiers() & GLUT_ACTIVE_SHIFT) != 0;
    if (k == GLUT_KEY_LEFT) spLeft = true; else if (k == GLUT_KEY_RIGHT) spRight = true;
    else if (k == GLUT_KEY_UP) spUp = true; else if (k == GLUT_KEY_DOWN) spDown = true;
}
static void specialUp(int k, int, int) {
    shiftHeld = (glutGetModifiers() & GLUT_ACTIVE_SHIFT) != 0;
    if (k == GLUT_KEY_LEFT) spLeft = false; else if (k == GLUT_KEY_RIGHT) spRight = false;
    else if (k == GLUT_KEY_UP) spUp = false; else if (k == GLUT_KEY_DOWN) spDown = false;
}
static void mouse(int b, int s, int x, int y) {
    if (b == GLUT_LEFT_BUTTON) { dragging = (s == GLUT_DOWN); lastMX = x; lastMY = y; }
    else if (s == GLUT_DOWN && b == 3) fovY = clampf(fovY - 3.0f, 30.0f, 85.0f);
    else if (s == GLUT_DOWN && b == 4) fovY = clampf(fovY + 3.0f, 30.0f, 85.0f);
}
static void motion(int x, int y) {
    if (!dragging) return;
    camYaw += (x - lastMX) * 0.2f; camPitch = clampf(camPitch - (y - lastMY) * 0.2f, -60.0f, 75.0f);
    lastMX = x; lastMY = y;
}

int main(int argc, char **argv) {
    glutInit(&argc, argv);
    unsigned int mode = GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH | GLUT_STENCIL;
    glutInitDisplayMode(mode | GLUT_MULTISAMPLE);                       // anti-aliasing if the GPU offers it
    if (!glutGet(GLUT_DISPLAY_MODE_POSSIBLE)) glutInitDisplayMode(mode);
    glutInitWindowSize(winW, winH); glutInitWindowPosition(0, 0);
    glutCreateWindow("3D Weather and Environment Visualization");

    glEnable(GL_DEPTH_TEST); glShadeModel(GL_SMOOTH);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST); glHint(GL_FOG_HINT, GL_NICEST);
    glHint(GL_POINT_SMOOTH_HINT, GL_NICEST); glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);

    buildLists();
    buildTextures();
    initScene();
    camPos.y = fmaxf(heightAt(camPos.x, camPos.z), 0.0f) + 1.9f;
    for (int i = 0; i < 60; ++i) update(0.05f);                         // settle initial state

    glutDisplayFunc(display); glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboard); glutKeyboardUpFunc(keyboardUp);
    glutSpecialFunc(special); glutSpecialUpFunc(specialUp);
    glutMouseFunc(mouse); glutMotionFunc(motion);
    glutIgnoreKeyRepeat(1);
    glutTimerFunc(8, timer, 0);
    glutMainLoop();
    return 0;
}
