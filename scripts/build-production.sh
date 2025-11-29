[ -d build ] && echo "Build directory already exists, skipping mkdir." || mkdir build
cd build && cmake -DCMAKE_BUILD_TYPE=Release -DINCLUDE_TESTS=OFF .. && make
