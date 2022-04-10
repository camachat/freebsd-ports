LLVM_VERSION?= 	${LLVM_DEFAULT}
BUILD_DEPENDS+= clang${LLVM_VERSION}:devel/llvm${LLVM_VERSION}
BINARY_ALIAS+= 	cpp=${LOCALBASE}/bin/clang-cpp${LLVM_VERSION} \
                cc=${LOCALBASE}/bin/clang${LLVM_VERSION} \
                c++=${LOCALBASE}/bin/clang++${LLVM_VERSION} \
                ar=${LOCALBASE}/bin/llvm-ar${LLVM_VERSION} \
                nm=${LOCALBASE}/bin/llvm-nm${LLVM_VERSION} \
                ld=${LOCALBASE}/bin/ld.lld${LLVM_VERSION}
