echo "Tests are built only for Debug and Release build types. If you built for Production, no tests are avaiable!"
ctest --test-dir build/src/lib
