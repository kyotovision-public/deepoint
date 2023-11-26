#version 330
out vec4 outputColor;
in vec2 UV;

uniform sampler2D iChannel0;
uniform vec3 iResolution;
vec4 iMouse = vec4(0);
uniform float iTime = 0.0;
uniform float xpos = 100.0;
uniform float ypos = 0.1;
uniform float size = 10.0;
uniform float arrow_length_mul = 1.0;

uniform float vdir_x = -1.0;
uniform float vdir_y = -1.0;
uniform float vdir_z = 1.0;

uniform vec3 arrow_color = vec3(0.,0.,0.);


const vec3 X = vec3(1., 0., 0.);
const vec3 Y = vec3(0., 1., 0.);
const vec3 Z = vec3(0., 0., 1.);

// YOURS

const float Z_NEAR = 1.0;
const float Z_FAR  = 400.0;

const float EPSILON = 0.01;

const float FOCAL_LENGTH = 30.0;
const vec3 EYE_LOOK_POINT = vec3(0, 0, 5);

const vec3 WHITE = vec3(1, 1, 1);
const vec3 BLACK = vec3(0, 0, 0);
const vec3 RED =   vec3(1, 0, 0);
const vec3 GREEN = vec3(0, 1, 0);
const vec3 BLUE =  vec3(0, 0, 1);

const vec3 TOP_BG_COLOR = WHITE;
const vec3 BOT_BG_COLOR = GREEN;

const vec3 AMBIENT_COLOR = WHITE;
const vec3 SPECULAR_COLOR = WHITE;

const float AMBIENT_RATIO = 0.37;
const float DIFFUSE_RATIO = 0.53;
const float SPECULAR_RATIO = 0.04;
const float SPECULAR_ALPHA = 2.0;

const vec3 LIGHT_DIRECTION = normalize(vec3(1, -1, -1));

vec2 normalizeAndCenter(in vec2 coord) {
    return (2.0 * coord - iResolution.xy) / iResolution.y;
}

vec3 rayDirection(vec3 eye, vec2 uv) {    
    vec3 z = normalize(eye - EYE_LOOK_POINT);
    vec3 x = normalize(cross(Y, z));
    vec3 y = cross(z, x);

    return normalize(
        x * uv.x 
        + y * uv.y 
        - z * FOCAL_LENGTH);
}

//
// Rotations
//

vec3 rotX(vec3 point, float angle) {
    mat3 matRotX = mat3(
        1.0, 0.0, 0.0, 
        0.0, cos(angle), -sin(angle), 
        0.0, sin(angle), cos(angle));
    return matRotX * point;
}

vec3 rotY(vec3 point, float angle) {
    mat3 matRotY = mat3( 
        cos(angle*0.5), 0.0, -sin(angle*0.5),
        0.0, 1.0, 0.0, 
        sin(angle*0.5), 0.0, cos(angle*0.5));
    return matRotY * point;
}

vec3 rotZ(vec3 point, float angle) {
    mat3 matRotZ = mat3(
        cos(angle*0.1), -sin(angle*0.1), 0.0, 
        sin(angle*0.1), cos(angle*0.1), 0.0,
        0.0, 0.0, 1.0
    );
    return matRotZ * point;
}

//
// Positioning
//

vec3 randomOrtho(vec3 v) {
    if (v.x != 0. || v.y != 0.) {
        return normalize(vec3(v.y, -v.x, 0.));
    } else {
        return normalize(vec3(0., v.z, -v.y));
    }
} 

vec3 atPosition(vec3 point, vec3 position) {
    return (point - position);
}

vec3 atCoordSystem(vec3 point, vec3 center, vec3 dx, vec3 dy, vec3 dz) {
    vec3 localPoint = (point - center);
    return vec3(
        dot(localPoint, dx),
        dot(localPoint, dy),
        dot(localPoint, dz));
}

vec3 atCoordSystemX(vec3 point, vec3 center, vec3 dx) {
    vec3 dy = randomOrtho(dx);
    vec3 dz = cross(dx, dy);

    return atCoordSystem(point, center, dx, dy, dz);
}

vec3 atCoordSystemY(vec3 point, vec3 center, vec3 dy) {
    vec3 dz = randomOrtho(dy);
    vec3 dx = cross(dy, dz);

    return atCoordSystem(point, center, dx, dy, dz);
}

vec3 atCoordSystemZ(vec3 point, vec3 center, vec3 dz) {
    vec3 dx = randomOrtho(dz);
    vec3 dy = cross(dz, dx);

    return atCoordSystem(point, center, dx, dy, dz);
}

//
// Shapes
//

float capsule(vec3 coord, float height, float radius)
{
    coord.y -= clamp( coord.y, 0.0, height );
    return length( coord ) - radius;
}

float roundCone(vec3 coord, in float radiusTop, float radiusBot, float height)
{
    vec2 q = vec2( length(coord.xz), coord.y );

    float b = (radiusBot-radiusTop)/height;
    float a = sqrt(1.0-b*b);
    float k = dot(q,vec2(-b,a));

    if( k < 0.0 ) return length(q) - radiusBot;
    if( k > a*height ) return length(q-vec2(0.0,height)) - radiusTop;

    return dot(q, vec2(a,b) ) - radiusBot;
}

//
// Boolean ops
//

vec4 shape(float dist, vec3 color) {
    return vec4(color, dist);
}

vec4 join(vec4 shape1, vec4 shape2) {
    if (shape1.a < shape2.a) {
        return shape1;
    } else {
        return shape2;
    }
}

vec4 join(vec4 shape1, vec4 shape2, vec4 shape3) {
    return join(join(shape1, shape2), shape3);
}

vec4 join(vec4 shape1, vec4 shape2, vec4 shape3, vec4 shape4) {
    return join(join(shape1, shape2, shape3), shape4);
}



//
// Scene
// x range: 355
// y rangeL 205



vec4 dist(in vec3 coord) {
    vec3 V_y = normalize(vec3(vdir_x,vdir_y,vdir_z));

    vec3 V_x = randomOrtho(V_y);

    vec3 V_z = cross(V_x,V_y);

    vec3 pos = vec3(xpos,ypos,0);

    vec3 offset = pos+V_y*size*arrow_length_mul*4.0;
    vec3 ARM_2_X = V_x;
    vec3 ARM_2_Y = V_y;
    vec3 ARM_2_Z = V_z;
    vec3 NOZE_O = vec3(0,11,0);
    vec3 NOZE_X = -V_x;
    vec3 NOZE_Y = V_y;
    vec3 NOZE_Z = V_z;

    vec4 noze = shape(roundCone(atCoordSystem(coord, offset, NOZE_X, NOZE_Y, NOZE_Z), size*0.05, size*0.4, size*arrow_length_mul*2.0), arrow_color);
    vec4 rightArm = shape(capsule(atCoordSystem(coord, pos, ARM_2_X, ARM_2_Y, ARM_2_Z), size*arrow_length_mul*5., size*0.1), arrow_color);
    return join(noze, rightArm);
}

//
//
//

bool rayMarching(in vec3 startPoint, in vec3 direction, out vec3 lastPoint, out vec3 color) {
    lastPoint = startPoint;
    for (int i = 0; i < 50; ++i) {
        vec4 d = dist(lastPoint);
        if (d.a < EPSILON) {
            color = d.xyz;
            return true;
        } else {
            lastPoint += d.a * direction;
        }
        if (lastPoint.z < -Z_FAR) {
            break;
        }
    }    
    return false;
}

vec3 norm(in vec3 coord) {
    vec3 eps = vec3( EPSILON, 0.0, 0.0 );
    vec3 nor = vec3(
        dist(coord+eps.xyy).a - dist(coord-eps.xyy).a,
        dist(coord+eps.yxy).a - dist(coord-eps.yxy).a,
        dist(coord+eps.yyx).a - dist(coord-eps.yyx).a);
    return normalize(nor);
}


vec3 cellShadingObjColor(vec3 point, vec3 ray, vec3 objColor) {
    vec3 n = norm(point);

    float diffuseValue = max(dot(-LIGHT_DIRECTION, n)/2.5+0.6, 0.);
    float specularValue = pow(max(dot(-reflect(LIGHT_DIRECTION, n), ray), 0.), SPECULAR_ALPHA);
    return (0.5*objColor + 0.5*AMBIENT_COLOR) * AMBIENT_RATIO
    + objColor * DIFFUSE_RATIO * diffuseValue
    + SPECULAR_COLOR * SPECULAR_RATIO * specularValue;
}


vec3 computeColor(vec2 fragCoord) {
    vec2 uv = normalizeAndCenter(fragCoord);
    vec3 eye = vec3(0, 0, 20);

    vec3 ray = rayDirection(eye, uv);

    vec3 intersection;
    vec3 color;
    bool intersected = rayMarching(eye, ray, intersection, color);
    if (intersected) {
        return cellShadingObjColor(intersection, ray, color);
    } else {
        return vec3(0,0,0);
    }
}

//#define SUPERSAMPLING
void mainImage( out vec4 fragColor, in vec2 fragCoord )
{    
#ifdef SUPERSAMPLING
    fragColor = vec4(0);
    float count = 0.;
    for(float dx=-0.5; dx<=0.5; dx+=0.5) {
        for(float dy=-0.5; dy<=0.5; dy+=0.5) {
            fragColor += vec4(computeColor(fragCoord + vec2(dx, dy)), 1.0);
            count += 1.;
        }
    }

    fragColor /= count;

#else
    fragColor = vec4(computeColor(fragCoord),1.0);
#endif
} 



void main()
{
    mainImage(outputColor, UV*iResolution.xy);
}
