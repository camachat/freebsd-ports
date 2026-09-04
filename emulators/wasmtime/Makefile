MASTERDIR=	${.CURDIR:H:H}/devel/libwasmtime
CATEGORIES=	emulators

PKGNAMEPREFIX=
WASMTIME_SLAVE=	cli
CARGO_BUILD_ARGS=	-p wasmtime-cli

FILESDIR=	${.CURDIR}/files
PLIST=	${.CURDIR}/pkg-plist
SUB_FILES=	binmisc-wasmtime-wrapper
SUB_LIST=	NAME=wasmtime WASMTIME_CLI=wasmtime
USE_RC_SUBR=	wasmtime

post-install:
	${INSTALL_SCRIPT} ${WRKDIR}/binmisc-wasmtime-wrapper \
		${STAGEDIR}${PREFIX}/bin/

.include "${MASTERDIR}/Makefile"
