PORTNAME=	emilua
DISTVERSIONPREFIX=	v
DISTVERSION=	0.7.2
CATEGORIES=	lang devel

MAINTAINER=	manipuladordedados@gmail.com
COMMENT=	Lua execution engine
WWW=		https://emilua.org/

LICENSE=	BSL
LICENSE_FILE=	${WRKSRC}/LICENSE_1_0.txt

IGNORE_FreeBSD_12=	not supported by Emilua
IGNORE_FreeBSD_13=	not supported by Emilua

BUILD_DEPENDS=	asciidoctor:textproc/rubygem-asciidoctor \
		boost-libs>0:devel/boost-libs \
		cereal>0:devel/cereal \
		cmake:devel/cmake-core \
		gawk:lang/gawk \
		gperf:devel/gperf \
		ksh93:shells/ksh \
		libfmt>0:devel/libfmt \
		re2c:devel/re2c

LIB_DEPENDS=	libserd-0.so:devel/serd \
		libsord-0.so:devel/sord

USES=		localbase luajit:luajit-openresty meson ncurses pathfix \
		pkgconfig ssl
USE_GCC=	12
USE_GITHUB=	nodefault
GH_TUPLE=	breese:trial.protocol:79149f6:trial_protocol/subprojects/trial-protocol
USE_GITLAB=	yes

CONFIGURE_ENV+=	BOOST_INCLUDEDIR="${LOCALBASE}/include" \
		BOOST_LIBRARYDIR="${LOCALBASE}/lib"

MESON_ARGS=	--buildtype=plain \
		--wrap-mode=nodownload \
		-Db_pch=false \
		-Denable_manpages=true \
		-Denable_tests=true

CXXFLAGS+=	"-DFMT_HEADER_ONLY"

post-extract:
	${CP} ${WRKSRC}/subprojects/packagefiles/trial.protocol/meson.build \
		${WRKSRC}/subprojects/trial-protocol/

.include <bsd.port.mk>
