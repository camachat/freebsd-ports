MASTERDIR=	${.CURDIR:H:H}/devel/libwasmtime
CATEGORIES=	emulators

PKGNAMEPREFIX=
WASMTIME_SLAVE=	cli
CARGO_CARGOTOML=	${WRKSRC}/Cargo.toml
PLIST_FILES=	bin/wasmtime

FILESDIR=	${.CURDIR}/files
SUB_LIST=	NAME=wasmtime
USE_RC_SUBR=	wasmtime

.include	"${MASTERDIR}/Makefile"
