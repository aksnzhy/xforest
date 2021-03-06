//------------------------------------------------------------------------------
// Copyright (c) 2018 by contributors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//------------------------------------------------------------------------------

/*!
*  Copyright (c) 2018 by Contributors
* \file stl-util.h
* \brief This file contains facilities that enhance the STL.
*/
#ifndef XFOREST_BASE_STL_UTIL_H_
#define XFOREST_BASE_STL_UTIL_H_

/*!
* \brief Delete elements (in pointer type) in a STL container like
* vector, list, and deque.
*/
template <class Container>
void STLDeleteElementsAndClear(Container* c) {
  for (typename Container::iterator iter = c->begin();
       iter != c->end(); ++iter) {
    if (*iter != NULL) {
      delete *iter;
    }
  }
  c->clear();
}

/*!
* \brief Delete elements (in pointer type) in a STL associative container
* like map and unordered_map.
*/
template <class AssocContainer>
void STLDeleteValuesAndClear(AssocContainer* c) {
  for (typename AssocContainer::iterator iter = c->begin();
       iter != c->end(); ++iter) {
    if (iter->second != NULL) {
      delete iter->second;
    }
  }
  c->clear();
}

#endif  // XFOREST_BASE_STL_UTIL_H_
