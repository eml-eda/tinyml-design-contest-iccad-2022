/**
  ******************************************************************************
  * @file    network_data_params.c
  * @author  AST Embedded Analytics Research Platform
  * @date    Fri Sep 30 15:29:03 2022
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * Copyright (c) 2022 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */

#include "network_data_params.h"


/**  Activations Section  ****************************************************/
ai_handle g_network_activations_table[1 + 2] = {
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
  AI_HANDLE_PTR(NULL),
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
};




/**  Weights Section  ********************************************************/
AI_ALIGNED(32)
const ai_u64 s_network_weights_array_u64[231] = {
  0xc0977ac1c04ada66U, 0x40398a1d40342fd4U, 0x40be7c124056cd7fU, 0xc19bed6440e0f125U,
  0x40cf0a0d4194be92U, 0x4138ea18c1973846U, 0x3e7a5a4040f7e9d3U, 0xc026be5c3e8b5e90U,
  0xc0870755c0b4e02dU, 0xbe9b8a0a3e61313fU, 0xbf0a63b43d812ec7U, 0xbeef73b13ec96918U,
  0x3ef8fd80bf0e3c67U, 0xbeee1298bf0a6e09U, 0xbebc40ca3ed05384U, 0x3ee6a6efbf035f2aU,
  0xbe99a0f6bec2b207U, 0xbf0c8f463ec6cf05U, 0x3beb65253b74cc32U, 0x3bd89cd4bc33d999U,
  0x3b77528c3bb09924U, 0x3c0b951cbb14e50bU, 0xbb74ac6a3beaf349U, 0x3c495badba854412U,
  0xbbc3cc8abb70e34dU, 0x40200e733c569430U, 0xbfff9e7ebfa9efd0U, 0xbfcbc56c3e148334U,
  0xbf67de083ea9f347U, 0xbf9e9cc1be50ee22U, 0xbf03e072bf18f511U, 0x3db54b64be483807U,
  0xbd7cc3e5bba20429U, 0xbfdec5c5be96abb9U, 0x4118bc61be1ae830U, 0x3cb0b5b54054ed25U,
  0x3b8c613d3be96c71U, 0xbc357c71bce36d46U, 0xbc3c5e803cca224fU, 0xbef47063bda81c54U,
  0xbe5c8d623d9cc31cU, 0xbe3b5d5e3e42c55bU, 0xbe6a70253d1e7972U, 0x3f1458ddbef8e683U,
  0xbe15c34a3f945abdU, 0xbc485a88be6232cbU, 0xbb822342be4a6fedU, 0xbc87fc2dbe65001cU,
  0xbf2c1eaabe234637U, 0xbf4dc42f3fb8e58fU, 0xbf4d5ed93ea5320fU, 0x3e22717e3f1cedfeU,
  0xbf21f72b3fd94eaaU, 0xbe559b9b3fdf46bcU, 0xbf05541c3eddda0bU, 0x3f36267e3f47be3bU,
  0xbd9a3ad23fd77aa9U, 0xbd752fb83c6f16e7U, 0x3c504e853d6427d4U, 0x3c5ae98b3b1ada16U,
  0xbecbbc9bbca030c6U, 0xbed77b6a3f9c84efU, 0xbf14390a3ea6ccafU, 0x3f02e6ce3f28414fU,
  0xbf0ac27f3f9c073fU, 0x3da1e02d3fb8c73aU, 0xbcecd1293f5be09cU, 0x3f6309973f03aa8dU,
  0xbfa4b2ae3fc67670U, 0xbf6fe5983fd34aa3U, 0xbf10a12b3f04aabfU, 0x3f5fff853ee9f2edU,
  0xbcbc06a74010a1b2U, 0xbea17d943d83a6cfU, 0xbc82f9993e184bfcU, 0xbd216fec3c38573dU,
  0xbe046d983c7e670dU, 0xc06e3b37c019a11cU, 0xc0391780bf5ff158U, 0xc034ba8dc07d0078U,
  0xbc69afa5bf2ae277U, 0xbe6d58a4be5db13dU, 0xbdf8f131be04df07U, 0xbd6f9e30be9b1901U,
  0xbdba2c653ded967eU, 0xbdda6251bea11047U, 0xbe6031493dfdab6fU, 0xbe3319f7be34d354U,
  0x3c8057ecbd891536U, 0xbd7a17acbe5b3130U, 0xbe2b39ff3dac75beU, 0xbddf97cabdf0c852U,
  0xbd8958943e39ffb0U, 0xbe50ba0abe6d0547U, 0xbe854cb9bcaae2bbU, 0xbe8f5d7cbe5409ddU,
  0x3d48acf8bd96998cU, 0xbe744ebbbe45f696U, 0xbe3b178dbced0a20U, 0xbeab84ccbe9c46faU,
  0xbd0c768e3da55761U, 0xbe0fc514be54c4c6U, 0xbe238a303dcf1037U, 0xbe5b5e04beb1b1b8U,
  0x3cdb74353c72b559U, 0xbe15bdc0bdeca8a8U, 0xbe6586b43d4c71ccU, 0xbe710390be4fc4cfU,
  0xbc82df193de1807eU, 0xbe31ba0fbe931a0aU, 0xbe688a313d81c2daU, 0xbe90bcbdbe85b2e5U,
  0x3d86522bbd8dbdf7U, 0xbe46b270be1ed212U, 0xbea73647bd3ce6c5U, 0xbe8dd27dbe7eaddcU,
  0x3dab5d633c3c6fc4U, 0xbe7c5577bea8ea0bU, 0xbeb2b2f83df0be34U, 0xbe84c210bea2e46eU,
  0x3cd187fa3de46db2U, 0xbe4e335fbea6c4e5U, 0xbeb5edbb3c216decU, 0xbea9a1cabea60685U,
  0x3d9971523dc5f96eU, 0xbde4a0fbbdf61f76U, 0xbebb2119bdab7d34U, 0xbe7ccbddbe8fcebaU,
  0x3ccc95dc3d95e0e9U, 0xbe41ee12beb05822U, 0xbeb310a6bce85ca4U, 0xbea4c0febe331f70U,
  0xbdc913b3bdde3df1U, 0xbe636754be8fb8adU, 0xbe87ebe7bd835bd8U, 0xbea59d10be86a416U,
  0x3df3eb983e25f176U, 0xbe0e4d64be9b3c48U, 0xbe93b62d3db09eadU, 0xbe31344cbe399ffcU,
  0xbd49c6393e262e36U, 0xbe67ae67be8a8683U, 0xbee6170fbdeb2738U, 0xbeaf1efebde09671U,
  0x3dfd21e6be566867U, 0xbe7c05e1bee01d51U, 0xbe6e5e5dbd4d97b6U, 0xbeb6378fbe469fecU,
  0x3df716a9bd5fce40U, 0xbe2ef8fdbea52e4bU, 0xbe565ff03e426badU, 0xbece1dabbeb6d33eU,
  0xbd5b53f13e4cf9f4U, 0xbe9ebe5abea2508bU, 0xbec48664bdb17856U, 0xbeb99ec6bed65421U,
  0xbe3182d3bdaddf31U, 0xbe9e8a9abe606618U, 0xbe9a0b66be52a243U, 0xbe853c66bea4420cU,
  0x3e30ed01be66c092U, 0xbe435745bed47e1bU, 0xbeae986c3e20cd6fU, 0xbea6b7f3bedc8b51U,
  0xbca60b983d68e723U, 0xbe9efc31be8dc0b0U, 0xbe9cdd88bbbd1325U, 0xbe764a88be951895U,
  0x3cf86aa7bd3be064U, 0xbe82768abeafa40fU, 0xbe9f1389be2b5f18U, 0xbea2d54dbebcb517U,
  0x3e14fb7ebe631b08U, 0xbe91729ebecb80eaU, 0xbeb727073c3e85aaU, 0xbecc586abe65af88U,
  0x3dd53e49bd0f1ceeU, 0xbe9769dfbea8dc77U, 0xbe9414d63e8883b5U, 0xbeb82b21be9f8436U,
  0xbdfce2febc63f548U, 0xbe94a419be9171b1U, 0xbeadcf7bbe02f120U, 0xbebc03b3be3ad8dcU,
  0x3b33167ebe2037ebU, 0xbe7a8bf4be14e0b8U, 0xbe40e66cbddad42eU, 0xbec556a3beb4ea43U,
  0x3d780abebde5b38bU, 0xbe19f14abeb4ae16U, 0xbeb6d41e3ddf1945U, 0xbed0f651be84faabU,
  0x3cfde2393d8f4d76U, 0xbe2dedcebe80d4faU, 0xbee1ab01bd847863U, 0xbe98bc82bed46da3U,
  0xbc95753dbe302fedU, 0xbe9f67dbbed5ddceU, 0xbe910982be181c76U, 0xbe6bbf0dbeb378b8U,
  0xbda71519bd89e408U, 0xbe3cf4fdbed0c18dU, 0xbea27a79bd371015U, 0xbeb3e0a1becd26fcU,
  0x3dc402863db02c31U, 0xbe019dd9beb0a4efU, 0xbebdc64d3d2fe0b2U, 0xbe9aa7a0beff9295U,
  0xbbbbd7863b8454d6U, 0xbea06603be9b8231U, 0xbe686b78bdc45b32U, 0xbe6642f0bf1125a7U,
  0x3d8e9b51bda37a26U, 0xbe86a53fbe9299b8U, 0xbea82b7cbced5ed8U, 0xbeb7d5e6bec23619U,
  0x3dde32c9bd0f022fU, 0xbe6bfa0bbe4ef372U, 0xbedd74543c9197cfU, 0xbe74894ebe9f695dU,
  0x3b8a8e4ebc93e8eaU, 0xbe0e696ebecdaca8U, 0xbea09fb53d1fac18U, 0xbe69dd45be4a5f96U,
  0xbe2142da3cc240b0U, 0xbe6429dbbe73e5d3U, 0xbe5bda71bdbeadf6U, 0xbea4041fbe8727edU,
  0x413d61ecbdfc4569U, 0x40bc4b18c0b09d96U, 0xc0a9274540b064b9U,
};


ai_handle g_network_weights_table[1 + 2] = {
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
  AI_HANDLE_PTR(s_network_weights_array_u64),
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
};

