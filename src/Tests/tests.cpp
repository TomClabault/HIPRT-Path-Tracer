#include "Renderer/bvh.h"
#include "Renderer/flattened_bvh.h"
#include "Renderer/ray.h"
#include "Renderer/triangle.h"
#include "Scene/scene_parser.h"
#include "Tests/bvh_tests.h"
#include "Tests/tests.h"
#include "Utils/utils.h"

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
    Scene scene = SceneParser::parse_scene_file("data/OBJs/cornell_pbr.obj");

    std::vector<Triangle> triangles = scene.make_triangles();
    BVH bvh(&triangles);
    test_bvh(bvh);
    test_flattened_bvh(bvh);
}
