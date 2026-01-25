PORTNAME=	lkl
DISTVERSION=	6.6.g20260115
CATEGORIES=	filesystems
PKGNAMEPREFIX=	fusefs-

MAINTAINER=	cem@FreeBSD.org
COMMENT=	Full-featured Linux BTRFS, Ext4, XFS as a FUSE module
WWW=		https://github.com/lkl/linux/

LICENSE=	GPLv2

BUILD_DEPENDS=	gnubc>0:math/gnubc \
		gnustat:sysutils/coreutils \
		gsed:textproc/gsed

USES=		bison:build fuse gmake python:build shebangfix
SHEBANG_FILES=	../../arch/lkl/scripts/headers_install.py
BINARY_ALIAS=	sed=gsed \
		stat=gnustat
USE_BINUTILS=	yes:build
USE_GCC=	yes:build
USE_GITHUB=	yes
GH_PROJECT=	linux
GH_TAGNAME=	f6d6f7d3c9a454f40dac9c8c612cbc7761ca8ab2

ONLY_FOR_ARCHS=	amd64
ONLY_FOR_ARCHS_REASON=ld formats not available for other architectures

WRKSRC_SUBDIR=	tools/lkl
MAKE_JOBS_UNSAFE=yes
MAKE_ARGS+=	V=1
# Need ports versions of 'bc', 'ld' to build!
MAKE_ARGS+=	PATH="${LOCALBASE}/bin:${PATH}" CC="${CC}" HOSTCC="${CC}"
#CFLAGS+=	-Wno-incompatible-pointer-types -Wno-cast-function-type -Wno-stringop-truncation

#PLIST_FILES=	bin/lklfuse lib/liblkl.a
PLIST_FILES=	lib/liblkl.a

do-install:
	#${INSTALL_PROGRAM} ${WRKSRC}/lklfuse ${STAGEDIR}${PREFIX}/bin
	${INSTALL_DATA} ${WRKSRC}/liblkl.a ${STAGEDIR}${PREFIX}/lib

.include <bsd.port.mk>
