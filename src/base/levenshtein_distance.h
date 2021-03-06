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
* \file levenshtein_distance.h
* \brief This file defines the StrSimilar class, which can 
* be used to find similar strings.
*/
#ifndef XFOREST_BASE_LEVEN_SHTEIN_DISTANCE_H_
#define XFOREST_BASE_LEVEN_SHTEIN_DISTANCE_H_

#include <vector>
#include <string>

#include "src/base/common.h"

/*!
* \brief StrSimilar class is used to find the similar string from the
* target string list. We can use this class like this:
*
*    vector<string> list;
*    list.push_back("alex");
*    list.push_back("apple");
*    list.push_back("mac");
*    StrSimilar ss;
*    bool bo;
*
*    bo = ss.Find(std::string("alex"), list); 
*    bo = ss.Find(std::string("zz"), list);
*
*    std::string similar;
*    int distance = ss.FindSimilar("alexx", list, result);
*
* The distance will 1, which is the levenshtein distance
* between 'alexx' and 'alex'. Also, the result will be 'alex',
* which is the most similar string in the list compared
* with 'alexx'.
*/
class StrSimilar {
 public:
  /*!
  * brief StrSimilar constructor
  */
  StrSimilar() { }

  /*!
  * brief StrSimilar deconstructor
  */
  ~StrSimilar() { }

  /*!
  * \brief Find if str in target string list
  * \param str data string
  * \param list target data list
  * \return true (false) if str is (not) in target string list.
  */
  bool Find(const std::string& str,
            const std::vector<std::string>& list);

  /*!
  * \brief Find the most similar string from string list.
  * \param str data string
  * \param list target data list
  * \param result most similar string
  * \return the minimal levenshtein distance.
  */
  int FindSimilar(const std::string& str,
                  const std::vector<std::string>& list,
                  std::string& result);

 private:
  /*!
  * \brief Calculate levenshtein distance.
  * \param source source string
  * \param target target string
  * \return levenshtein distance
  */
  int ldistance(const std::string& source,
                const std::string& target);

  DISALLOW_COPY_AND_ASSIGN(StrSimilar);
};

#endif  // XFOREST_BASE_LEVEN_SHTEIN_DISTANCE_H_
