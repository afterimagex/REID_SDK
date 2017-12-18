// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_MODULES_CC_DRAGON_H_
#define DRAGON_MODULES_CC_DRAGON_H_

#include <string>
#include <cstdint>
#include <vector>

#ifdef WIN32
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT
#endif

namespace dragon {

typedef int64_t TIndex;

class Workspace;

class Device {
    enum DeviceType { CPU, CUDA };

 public:
    EXPORT Device();
    EXPORT explicit Device(std::string device_type);
    EXPORT Device(std::string device_type, int device_id);

    EXPORT const DeviceType& device_type() const { return device_type_; }
    EXPORT const int device_id() const { return device_id_; }

 private:
     DeviceType device_type_;
     int device_id_;
};

EXPORT Workspace* CreateWorkspace(const std::string& name);

EXPORT Workspace* ResetWorkspace(const std::string& name);

EXPORT void ReleaseWorkspace(const std::string& name);

EXPORT void MoveWorkspace(Workspace* main, Workspace* sub);

EXPORT std::string CreateGraph(const std::string& graph_file, Workspace* ws);

EXPORT std::string CreateGraph(const std::string& graph_file, const Device& device, Workspace* ws);

EXPORT void RunGraph(const std::string& graph_name, Workspace* ws);

EXPORT void CreateTensor(const std::string& name, Workspace* ws);

template <typename T>
void FeedTensor(const std::string& name,
                const std::vector<TIndex>& shape,
                const T* data,
                const Device& device,
                Workspace* ws);

template <typename T>
T* FetchTensor(const std::string& name,
               std::vector<TIndex>& shape,
               Workspace* ws);

template EXPORT float* FetchTensor(const std::string&,
                                   std::vector<TIndex>&,
                                   Workspace*);

template EXPORT void FeedTensor(const std::string&,
                                const std::vector<TIndex>&,
                                const float*, 
                                const Device&, 
                                Workspace*);

EXPORT void LoadCaffemodel(const std::string& model_file, Workspace* ws);

EXPORT void TransplantCaffeModel(const std::string& input_model, const std::string& output_model);

EXPORT void LoadDragonmodel(const std::string& model_file, Workspace* ws);

EXPORT void SetLogLevel(const std::string& level);

}    // namespace dragon

#endif    // DRAGON_MODULES_CC_DRAGON_H_