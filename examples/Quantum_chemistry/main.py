'''
 *
 * Copyright (C) 2020 Universitat Politècnica de Catalunya.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
'''

# -*- coding: utf-8 -*-

import sys
import tensorflow as tf
sys.path.append('./utils/')
import framework


def norm_mu(feature, feature_name):
    feature = (feature - 2.70426) / 1.52855
    return feature

def denorm_mu(feature, feature_name):
    feature = (feature * 1.52855) + 2.70426
    return feature


def main():
    model = framework.create_model()
    framework.debug(model)
    framework.train_and_evaluate(model)


if __name__ == "__main__":
        main ()