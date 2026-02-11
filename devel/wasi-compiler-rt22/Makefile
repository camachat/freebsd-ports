# must sync with devel/llvm22
DISTVERSION=	22.1.0-rc2
TRIPLE=		wasm32-wasip1
COMPILER_RT_OS_DIR=	wasm32-unknown-wasip1
PLIST_FILES=	${LLVM_PREFIX}/lib/clang/${LLVM_VERSION}/lib/${COMPILER_RT_OS_DIR}/libclang_rt.builtins.a

post-configure:
	${REINPLACE_CMD} -e 's|libclang_rt.builtins-wasm32|libclang_rt.builtins|g' \
		${CONFIGURE_WRKSRC}/cmake_install.cmake \
		${CONFIGURE_WRKSRC}/build.ninja

MASTERDIR=	${.CURDIR}/../wasi-compiler-rt

.include "${MASTERDIR}/Makefile"
