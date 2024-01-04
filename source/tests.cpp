#include "rapidobj.hpp"

#include "bvh.h"
#include "bvh_tests.h"
#include "flattened_bvh.h"
#include "ray.h"
#include "tests.h"
#include "triangle.h"

bool compare_points(const Point& a, const Point& b)
{
    return std::abs(a.x - b.x) < 1.0e-5f
            && std::abs(a.y - b.y) < 1.0e-5f && std::abs(a.z - b.z) < 1.0e-5f;
}

void test_bvh(BVH& bvh)
{
    std::cout << "[TESTS]: BVH MISSING RAYS --- ";
    for (const Ray& ray : bvh_test_rays_no_inter)
    {
        HitInfo closest_hit_info;
        if (bvh.intersect(ray, closest_hit_info))
        {
            std::cout << "Intersection found with the ray: " << ray << " but no intersection should have been found" << std::endl;
            std::cout << "Intersection t/point: " << closest_hit_info.t << "/" << ray.origin + closest_hit_info.t * ray.direction << std::endl;

            std::exit(-1);
        }
    }
    std::cout << "OK" << std::endl;

    std::cout << "[TESTS]: BVH HITTING RAYS --- ";
    for (int index = 0; index < bvh_test_rays_inter.size(); index++)
    {
        const Ray& ray = bvh_test_rays_inter[index];
        const Point& expected_point = bvh_test_rays_inter_result_points[index];

        HitInfo closest_hit_info;
        if (!bvh.intersect(ray, closest_hit_info))
        {
            std::cout << "No intersection was found with the ray: " << ray << " but an intersection should have been found" << std::endl;
            std::cout << "Expected intersection: " << expected_point << std::endl;

            std::exit(-1);
        }
        else
        {
            Point inter_point = ray.origin + closest_hit_info.t * ray.direction;
            if (!compare_points(inter_point, expected_point))
            {
                std::cout << "An intersection was found but the intersection is incorrect. Found point: " << inter_point << " | expected point: " << expected_point << std::endl;

                std::exit(-1);
            }
        }
    }
    std::cout << "OK" << std::endl;
}

int small_flat_bvh_tests()
{
    Ray ray(Point(0, 0, 0), Vector(0, 0, -1));

    std::vector<Triangle> triangles_tests;
    triangles_tests.push_back(Triangle(Point(0, 0, -2), Point(2, 0, -2), Point(1, 1, -2)));
    triangles_tests.push_back(Triangle(Point(0, 0, -3), Point(2, 0, -3), Point(1, 1, -3)));
    triangles_tests.push_back(Triangle(Point(0, 0, -4), Point(2, 0, -4), Point(1, 1, -4)));
    triangles_tests.push_back(Triangle(Point(0, 0, -5), Point(2, 0, -5), Point(1, 1, -5)));
    triangles_tests.push_back(Triangle(Point(0, 0, -6), Point(2, 0, -6), Point(1, 1, -6)));
    triangles_tests.push_back(Triangle(Point(-2, 0, -2), Point(0, 0, -2), Point(-1, 1, -2)));
    triangles_tests.push_back(Triangle(Point(2, 0, -3), Point(4, 0, -3), Point(3, 1, -3)));
    triangles_tests.push_back(Triangle(Point(0, -2, -4), Point(2, -2, -4), Point(1, -1, -4)));
    triangles_tests.push_back(Triangle(Point(0, -2, -5), Point(2, -2, -5), Point(1, -1, -5)));

    BVH bvh(&triangles_tests);

    FlattenedBVH flat = bvh.flatten();

    HitInfo hit_info;
    if (!flat.intersect(ray, hit_info, triangles_tests))
    {
        std::cout << "No intersection was found with the ray: " << ray << " but an intersection should have been found" << std::endl;
        std::cout << "Expected intersection: " << Point(0, 0, -2) << std::endl;

        return -1;
    }
    else
    {
        Point inter_point = hit_info.t * ray.direction + ray.origin;

        if (!compare_points(inter_point, Point(0, 0, -2)))
        {
            std::cout << "An intersection was found with the flat BVH but it was incorrect" << std::endl;
            std::cout << "Expected intersection vs found: " << Point(0, 0, -2) << " vs " << hit_info.t * ray.direction + ray.origin << std::endl;

            return -1;
        }
    }

    return 0;
}

void test_flattened_bvh(BVH& bvh)
{
    FlattenedBVH flat_bvh = bvh.flatten();

    std::cout << "[TESTS]: FLATTENED BVH MISSING RAYS --- ";
    for (const Ray& ray : bvh_test_rays_no_inter)
    {
        HitInfo closest_hit_info;
        if (flat_bvh.intersect(ray, closest_hit_info, *bvh._triangles))
        {
            std::cout << "Intersection found with the ray: " << ray << " but no intersection should have been found" << std::endl;
            std::cout << "Intersection t/point: " << closest_hit_info.t << "/" << ray.origin + closest_hit_info.t * ray.direction << std::endl;

            std::exit(-1);
        }
    }
    std::cout << "OK" << std::endl;

    std::cout << "[TESTS]: FLATTENED BVH HITTING RAYS --- ";
    if(small_flat_bvh_tests() != 0)
        std::exit(-1);

    for (int index = 0; index < bvh_test_rays_inter.size(); index++)
    {
        const Ray& ray = bvh_test_rays_inter[index];
        const Point& expected_point = bvh_test_rays_inter_result_points[index];

        HitInfo closest_hit_info;
        if (!flat_bvh.intersect(ray, closest_hit_info, *bvh._triangles))
        {
            std::cout << "No intersection was found with the ray: " << ray << " but an intersection should have been found" << std::endl;
            std::cout << "Expected intersection: " << expected_point << std::endl;

            std::exit(-1);
        }
        else
        {
            Point inter_point = ray.origin + closest_hit_info.t * ray.direction;
            if (!compare_points(inter_point, expected_point))
            {
                std::cout << "An intersection was found but the intersection is incorrect. Found point: " << inter_point << " | expected point: " << expected_point << std::endl;

                std::exit(-1);
            }
        }
    }


    std::cout << "OK" << std::endl;
}

void regression_tests()
{
    rapidobj::Result parsed_obj = rapidobj::ParseFile("../SYCL-ray-tracing/data/OBJs/cornell_pbr.obj", rapidobj::MaterialLibrary::Default());
    if (parsed_obj.error)
    {
        std::cout << "There was an error loading the OBJ file: " << parsed_obj.error.code.message() << std::endl;
        std::cin.get();

        std::exit(-1);
    }
    rapidobj::Triangulate(parsed_obj);

    const rapidobj::Array<float>& positions = parsed_obj.attributes.positions;
    std::vector<Triangle> triangle_host_buffer;
    for (rapidobj::Shape& shape : parsed_obj.shapes)
    {
        rapidobj::Mesh& mesh = shape.mesh;
        for (int i = 0; i < mesh.indices.size(); i += 3)
        {
            int index_0 = mesh.indices[i + 0].position_index;
            int index_1 = mesh.indices[i + 1].position_index;
            int index_2 = mesh.indices[i + 2].position_index;

            Point A = Point(positions[index_0 * 3 + 0], positions[index_0 * 3 + 1], positions[index_0 * 3 + 2]);
            Point B = Point(positions[index_1 * 3 + 0], positions[index_1 * 3 + 1], positions[index_1 * 3 + 2]);
            Point C = Point(positions[index_2 * 3 + 0], positions[index_2 * 3 + 1], positions[index_2 * 3 + 2]);

            Triangle triangle(A, B, C);
            triangle_host_buffer.push_back(triangle);
        }
    }

    BVH bvh(&triangle_host_buffer);
    test_bvh(bvh);
    test_flattened_bvh(bvh);
}
