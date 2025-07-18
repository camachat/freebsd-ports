PORTNAME=	xmrig
DISTVERSIONPREFIX=	v
DISTVERSION=	6.24.0
CATEGORIES=	net-p2p

MAINTAINER=	ehaupt@FreeBSD.org
COMMENT=	High performance Monero (XMR) CPU miner
WWW=		https://github.com/xmrig/xmrig

LICENSE=	GPLv3
LICENSE_FILE=	${WRKSRC}/LICENSE

BROKEN_powerpc64=	fails to compile: g++9: error: unrecognized command line option '-maes'; did you mean '-mads'?
BROKEN_riscv64=	fails to compile: /nxb-bin/usr/lib/clang/11.0.0/include/cpuid.h:11:2: error: this header is for x86 only

USES=		cmake compiler:c++11-lang ssl
USE_GITHUB=	yes
USE_RC_SUBR=	${PORTNAME}

SUB_FILES=	pkg-message
SUB_LIST=	PORTNAME=${PORTNAME} \
		USERS=${USERS}
USERS=		xmrig

OPTIONS_DEFINE=		HWLOC MICROHTTPD STATIC
OPTIONS_DEFAULT=	HWLOC MICROHTTPD
HWLOC_DESC=		Portable hardware locality
MICROHTTPD_DESC=	Build with libmicrohttpd support

HWLOC_LIB_DEPENDS+=		libhwloc.so.15:devel/hwloc2
HWLOC_CMAKE_OFF=		-DWITH_HWLOC=OFF
MICROHTTPD_LIB_DEPENDS+=	libmicrohttpd.so:www/libmicrohttpd \
				libuv.so:devel/libuv
MICROHTTPD_CMAKE_OFF=		-DWITH_HTTPD=OFF

STATIC_PREVENTS=	HWLOC MICROHTTPD
STATIC_PREVENTS_MSG=	hwloc and libmicrohttpd support cannot be built static

STATIC_BUILD_DEPENDS+=	${LOCALBASE}/lib/libuv.a:devel/libuv
STATIC_CMAKE_ON=	-DBUILD_STATIC=ON

do-install:
	${INSTALL_PROGRAM} ${WRKDIR}/.build/${PORTNAME} ${STAGEDIR}${PREFIX}/bin
	@${MKDIR} ${STAGEDIR}${ETCDIR}
	${INSTALL_DATA} ${WRKSRC}/src/config.json ${STAGEDIR}${ETCDIR}/config.json.sample

.include <bsd.port.mk>
