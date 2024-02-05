export PATH=$PATH:/home/victor/mlir_ws/
# /home/victor/mlir_ws/cmake-3.28.1-linux-x86_64/bin/cmake -G Ninja ../llvm \
#    -DLLVM_ENABLE_PROJECTS=mlir \
#    -DLLVM_BUILD_EXAMPLES=ON \
#    -DCMAKE_BUILD_TYPE=Release \
#    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON 
cd build 
/home/victor/mlir_ws/cmake-3.28.1-linux-x86_64/bin/cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="Native" \
   -DCMAKE_BUILD_TYPE=Release \
   -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
   -DLLVM_ENABLE_ASSERTIONS=ON
