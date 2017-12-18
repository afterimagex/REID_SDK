// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_CORE_MODEL_H_
#define DRAGON_CORE_MODEL_H_

#include <string>

#include "dragon.h"
#include "blob.h"

class Model {
 public:
     Model(const std::string graph_file, 
           const dragon::Device& device, 
           dragon::Workspace* ws)
         : graph_file_(graph_file), device_(device), ws_(ws) {}

    virtual void CreateVariables() = 0;
    virtual void FeedVariables() = 0;

    inline void Init() {
        CreateVariables();
        graph_name_ = CreateGraph(graph_file_, device_, ws_);
    }

    inline void Run() {
        FeedVariables();
        dragon::RunGraph(graph_name_, ws_);
    }

    inline void CreateVariable(const std::string& name) { 
        dragon::CreateTensor(name, ws_); 
    }

    template <typename T>
    inline void FeedVariable(const std::string& name, Blob<T>& blob) {
        dragon::FeedTensor(name, blob.shape(), blob.data(), device_, ws_);
    }

    template <typename T>
    inline void FetchVariable(const std::string& name, Blob<T>& blob) {
        std::vector<TIndex> shape;
        T* data = dragon::FetchTensor<T>(name, shape, ws_);
        blob.Assign(shape, data);
    }

    inline dragon::Workspace* ws() const { return ws_; }

 protected:
    std::string graph_file_, graph_name_;
    dragon::Workspace* ws_;
    dragon::Device device_;
};

#endif    // DRAGON_CORE_MODEL_H_