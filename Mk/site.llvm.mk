.if !defined(_SITE_LLVM_MK_INCLUDED)
_SITE_LLVM_MK_INCLUDED=	site.llvm.mk

LLVM_VERSION?= 	${LLVM_DEFAULT}

BUILD_DEPENDS+= clang${LLVM_VERSION}:devel/llvm${LLVM_VERSION}

BINARY_ALIAS+= 	cpp=${LOCALBASE}/bin/clang-cpp${LLVM_VERSION} \
                cc=${LOCALBASE}/bin/clang${LLVM_VERSION} \
                c++=${LOCALBASE}/bin/clang++${LLVM_VERSION} \
                ar=${LOCALBASE}/bin/llvm-ar${LLVM_VERSION} \
                nm=${LOCALBASE}/bin/llvm-nm${LLVM_VERSION} \
                ld=${LOCALBASE}/bin/ld.lld${LLVM_VERSION} \
                objcopy=${LOCALBASE}/bin/llvm-objcopy${LLVM_VERSION} \
                objdump=${LOCALBASE}/bin/llvm-objdump${LLVM_VERSION} \
                readobj=${LOCALBASE}/bin/llvm-readobj${LLVM_VERSION} \
                ranlib=${LOCALBASE}/bin/llvm-ranlib${LLVM_VERSION} \
                readelf=${LOCALBASE}/bin/llvm-readelf${LLVM_VERSION} \
                size=${LOCALBASE}/bin/llvm-size${LLVM_VERSION} \
                strings=${LOCALBASE}/bin/llvm-strings${LLVM_VERSION} \
                strip=${LOCALBASE}/bin/llvm-strip${LLVM_VERSION}

CPP     = ${LOCALBASE}/bin/clang-cpp${LLVM_VERSION}
CC      = ${LOCALBASE}/bin/clang${LLVM_VERSION}
CXX     = ${LOCALBASE}/bin/clang++${LLVM_VERSION}
AR      = ${LOCALBASE}/bin/llvm-ar${LLVM_VERSION}
NM      = ${LOCALBASE}/bin/llvm-nm${LLVM_VERSION}
LD      = ${LOCALBASE}/bin/ld.lld${LLVM_VERSION}
OBJCOPY = ${LOCALBASE}/bin/llvm-objcopy${LLVM_VERSION}
OBJDUMP = ${LOCALBASE}/bin/llvm-objdump${LLVM_VERSION}
READOBJ = ${LOCALBASE}/bin/llvm-readobj${LLVM_VERSION}
RANLIB  = ${LOCALBASE}/bin/llvm-ranlib${LLVM_VERSION}
READELF = ${LOCALBASE}/bin/llvm-readelf${LLVM_VERSION}
SIZE    = ${LOCALBASE}/bin/llvm-size${LLVM_VERSION}
STRINGS = ${LOCALBASE}/bin/llvm-strings${LLVM_VERSION}
STRIP   = ${LOCALBASE}/bin/llvm-strip${LLVM_VERSION}
.endif # _SITE_LLVM_MK_INCLUDED

