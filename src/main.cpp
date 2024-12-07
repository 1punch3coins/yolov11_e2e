#include <iostream>
#include <chrono>
#if __GNUC__ > 8
#include <filesystem>
namespace fs = std::filesystem;
#else 
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

#include "algos/yolov11_e2e.hpp"

void test_yolov11(const std::string& input_pwd, const std::string& otuput_pwd, const std::string& model_pwd) {
    Yolov11E2e yolo;
    cv::Mat ori_img = cv::imread(input_pwd);
    if (yolo.Initialize(model_pwd) == 1) {
        std::cerr << "yolov11 initialization uncompleted" << std::endl;
        return;
    }
    Yolov11E2e::Result det_res;
    yolo.Process(ori_img, det_res);
    // while(true){yolo.Process(ori_img, det_res);}
    for (const auto& box: det_res.batched_bbox_vec[0]) {
        cv::putText(ori_img, box.cls_name, cv::Point(box.x, box.y - 6), 0, 0.8, cv::Scalar(0, 255, 0), 2);
        cv::rectangle(ori_img, cv::Rect(box.x, box.y, box.w, box.h), cv::Scalar(0, 255, 0), 2);
    }
    cv::imwrite(otuput_pwd, ori_img);
    yolo.Finalize();
}

void test_yolov11(const std::vector<std::string>& input_pwds, const std::vector<std::string>& otuput_pwds, const std::string& model_pwd) {
    assert(input_pwds.size() == otuput_pwds.size());
    Yolov11E2e yolo;
    if (yolo.Initialize(model_pwd) == 1) {
        std::cerr << "yolov11 initialization uncompleted" << std::endl;
        return;
    }
    Yolov11E2e::Result det_res;
    
    std::vector<cv::Mat> cv_mats;
    cv_mats.reserve(input_pwds.size());
    for (unsigned i = 0; i < input_pwds.size(); i++) {
        cv_mats.push_back(cv::imread(input_pwds[i]));
    }
    yolo.Process(cv_mats, det_res);
    // while(true) {yolo.Process(cv_mats, det_res);}
    for (unsigned i = 0; i < input_pwds.size(); i++) {
        for (const auto& box: det_res.batched_bbox_vec[i]) {
            cv::putText(cv_mats[i], box.cls_name, cv::Point(box.x, box.y - 6), 0, 0.8, cv::Scalar(0, 255, 0), 2);
            cv::rectangle(cv_mats[i], cv::Rect(box.x, box.y, box.w, box.h), cv::Scalar(0, 255, 0), 2);
        }
        cv::imwrite(otuput_pwds[i], cv_mats[i]);
    }
    yolo.Finalize();
}

void get_in_img_paths(const char *folder_dir, std::vector<std::string>& file_paths, const char *suffix = ".jpg") {
    if (fs::is_directory(folder_dir)) {
        for (const auto& entry: fs::directory_iterator(folder_dir)) {
            if (entry.path().extension() == suffix) {
                file_paths.push_back(entry.path());
            }
        }
    } else {
        printf("directory %s is not accessible\n", folder_dir);
        exit(EXIT_FAILURE);
    }
}

void set_out_img_paths(const char *folder_dir, const std::vector<std::string>& in_file_paths, std::vector<std::string>& out_file_paths) {
    if (fs::is_directory(folder_dir)) {
        for (const auto& s: in_file_paths) {
            auto out_file_path = fs::path(folder_dir);
            auto in_file_path = fs::path(s);
            out_file_path.append((in_file_path.stem().concat("_res")+=in_file_path.extension()).c_str());
            out_file_paths.push_back(out_file_path);
        }
    } else {
        printf("directory %s is not accessible\n", folder_dir);
        exit(EXIT_FAILURE);
    }
}

static void help() {
    printf(
        "Usage: \n"
        "    ./yolo11 in/ out/ \n"
        "    Run yolo11 inference with .jpg files under in, save .jpg under out\n"
    );
    exit(EXIT_FAILURE);
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        help();
    }
    const char* model_pwd = argv[1];
    const char* in_dir    = argv[2];
    const char* out_dir   = argv[3];
    std::vector<std::string> input_pwds, output_pwds;
    get_in_img_paths(in_dir, input_pwds);
    set_out_img_paths(out_dir, input_pwds, output_pwds);
    test_yolov11(input_pwds, output_pwds, model_pwd);
    return 0;
}