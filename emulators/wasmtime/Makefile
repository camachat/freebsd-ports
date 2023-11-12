MASTERDIR=	${.CURDIR:H:H}/devel/libwasmtime
CATEGORIES=	emulators

PKGNAMEPREFIX=
WASMTIME_SLAVE=	cli
CARGO_CARGOTOML=	${WRKSRC}/Cargo.toml
PLIST_FILES=	bin/wasmtime
#PLIST_FILES=	bin/wasmtime bin/binmisc-wasmtime-wrapper

FILESDIR=	${.CURDIR}/files
SUB_FILES=	binmisc-wasmtime-wrapper
SUB_LIST=	NAME=wasmtime WASMTIME_CLI=wasmtime
USE_RC_SUBR=	wasmtime

post-install:
	${INSTALL_SCRIPT} ${WRKDIR}/binmisc-wasmtime-wrapper \
		${STAGEDIR}${PREFIX}/bin/

.include "${MASTERDIR}/Makefile"
