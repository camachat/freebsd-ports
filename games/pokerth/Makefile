PORTNAME=	pokerth
PORTVERSION=	1.1.2
PORTREVISION=	27
CATEGORIES=	games
MASTER_SITES=	SF

PATCH_SITES=	https://gitweb.gentoo.org/repo/gentoo.git/plain/games-board/pokerth/files/ \
		https://raw.githubusercontent.com/gentoo/gentoo/master/games-board/pokerth/files/
PATCHFILES=	pokerth-1.1.2-boost-1.73.patch:-p1 \
		pokerth-1.1.2-boost-1.85.patch:-p1

MAINTAINER=	ports@FreeBSD.org
COMMENT=	Poker game written in C++/Qt
WWW=		https://www.pokerth.net/

LICENSE=	AGPLv3+
LICENSE_FILE=	${WRKSRC}/COPYING

BROKEN=		fails to build with Boost>=1.87

LIB_DEPENDS=	libboost_thread.so:devel/boost-libs \
		libmikmod.so:audio/libmikmod \
		libgnutls.so:security/gnutls \
		libcurl.so:ftp/curl \
		libgsasl.so:security/libgsasl \
		libtinyxml.so:textproc/tinyxml \
		libprotobuf.so:devel/protobuf \
		libircclient.so:irc/libircclient
BUILD_DEPENDS=	${LOCALBASE}/include/websocketpp/client.hpp:devel/websocketpp \
		protoc:devel/protobuf

USES=		compiler:c++17-lang gl gmake iconv localbase:ldflags qmake qt:5 sdl sqlite ssl
USE_GL=		gl
USE_CXXSTD=	c++17
USE_QT=		buildtools:build core gui network sql sql-sqlite3 widgets
USE_SDL=	mixer

CXXFLAGS+=	-D_LIBCPP_ENABLE_CXX17_REMOVED_UNARY_BINARY_FUNCTION
LDFLAGS+=	-labsl_log_internal_check_op \
		-labsl_log_internal_message \
		-labsl_log_internal_nullguard \
		-lprotobuf
QMAKE_SOURCE_PATH=	pokerth.pro
QMAKE_ARGS+=	CONFIG+="client"
WRKSRC=		${WRKDIR}/${DISTNAME}-rc

PORTDOCS=	server_setup_howto.txt

OPTIONS_DEFINE=	DOCS

.include <bsd.port.options.mk>

post-patch:
	@${REINPLACE_CMD} -E \
		-e 's|/usr([a-z|/]*)|${LOCALBASE}/\1|g' \
		-e 's|(-lcurl)|\1 ${ICONV_LIB}|g' \
		-e 's|(-lcrypto) -liconv|\1 ${ICONV_LIB}|' \
		-e 's:boost_([a-z]+)-[-|a-z|0-9|_]*:boost_\1:g' \
		-e 's|LIB_DIRS =.*|LIB_DIRS = ${LOCALBASE}/lib|g' \
		-e '/.*QMAKE_CXXFLAGS.*/d' \
		${WRKSRC}/*.pro

pre-build:
	(cd ${WRKSRC} && \
		 protoc --cpp_out=src/third_party/protobuf chatcleaner.proto && \
		 protoc --cpp_out=src/third_party/protobuf pokerth.proto \
	)

post-install:
	@${STRIP_CMD} ${STAGEDIR}${PREFIX}/bin/pokerth
	${INSTALL_PROGRAM} ${WRKSRC}/bin/pokerth_server ${STAGEDIR}${PREFIX}/bin
	${INSTALL_MAN} ${WRKSRC}/docs/pokerth.1 ${STAGEDIR}${PREFIX}/share/man/man1
	@${MKDIR} ${STAGEDIR}${DOCSDIR}
	${INSTALL_DATA} ${WRKSRC}/docs/server_setup_howto.txt ${STAGEDIR}${DOCSDIR}

.include <bsd.port.mk>
