// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef LAYER_EINSUM_H
#define LAYER_EINSUM_H

//#include "layer.h"

#include "mat.h"
#include <vector>
namespace ncnn {
//class Einsum : public Layer
class Einsum
{
public:
    Einsum();

	virtual int load_param(const std::string& equation);
	virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs) const;
#if 0
    virtual int load_param(const ParamDict& pd);

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
#endif
public:
    // equation tokens
    std::vector<std::string> lhs_tokens;
    std::string rhs_token;
};

} // namespace ncnn

#endif // LAYER_EINSUM_H
