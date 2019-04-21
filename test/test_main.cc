//
// Created by lei on 4/21/19.
//

#define CATCH_CONFIG_RUNNER
#include "catch.hpp"


int main(int argc, char *argv[])
{
  int result = Catch::Session().run(argc, argv);
  return result;
}

