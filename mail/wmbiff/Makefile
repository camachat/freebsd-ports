# Created by: Jim Mock <jim@FreeBSD.org>
<<<<<<< HEAD
# $FreeBSD$

PORTNAME=	wmbiff
PORTVERSION=	0.4.27
PORTREVISION=	19
CATEGORIES=	mail windowmaker afterstep
MASTER_SITES=	SF
=======

PORTNAME=	wmbiff
PORTVERSION=	0.4.36
PORTREVISION=	1
CATEGORIES=	mail windowmaker afterstep
MASTER_SITES=	https://www.dockapps.net/download/
>>>>>>> upstream/main

MAINTAINER=	garga@FreeBSD.org
COMMENT=	xbiff-like Window Maker dock app

LICENSE=	GPLv2

<<<<<<< HEAD
USES=		shebangfix xorg

OPTIONS_DEFINE=	GNUTLS
SHEBANG_FILES=	scripts/security.debian.rb
USE_XORG=	x11 xext xpm
GNU_CONFIGURE=	yes

# Temporary fix to build with clang11
CFLAGS+=	-fcommon

PLIST_FILES=	bin/wmbiff \
		lib/wmbiff/security.debian.rb \
		man/man1/wmbiff.1.gz \
		man/man5/wmbiffrc.5.gz \
		${DATADIR}/sample.wmbiffrc \
		${DATADIR}/skins/wmbiff-master-contrast.xpm \
		${DATADIR}/skins/wmbiff-master-led.xpm

GNUTLS_LIB_DEPENDS=	libgcrypt.so:security/libgcrypt \
			libgnutls.so:security/gnutls
GNUTLS_CONFIGURE_ON=	--with-libgnutls-prefix=${LOCALBASE} \
			--with-libgcrypt-prefix=${LOCALBASE}
GNUTLS_CONFIGURE_OFF=	--disable-crypto

post-install:
	@${ECHO_MSG} "===>   Installing sample.wmbiffrc to ${DATADIR}"
	@${ECHO_MSG} "===>   Modify it to suit your needs and copy to ~/.wmbiffrc"
=======
USES=		pkgconfig xorg

USE_XORG=	x11 xext xpm
GNU_CONFIGURE=	yes
SUB_FILES=	pkg-message

OPTIONS_DEFINE=	GNUTLS
OPTIONS_DEFAULT=GNUTLS

GNUTLS_LIB_DEPENDS=	libgcrypt.so:security/libgcrypt \
			libgnutls.so:security/gnutls \
			libgpg-error.so:security/libgpg-error
GNUTLS_CONFIGURE_ON=	--with-libgcrypt-prefix=${LOCALBASE}
GNUTLS_CONFIGURE_OFF=	--disable-crypto

post-patch:
	@${REINPLACE_CMD} -e '/^SUBDIRS *=/ s/scripts//' \
		${WRKSRC}/Makefile.in
	@${REINPLACE_CMD} -e '/security.debian.rb/ s/^/# /' \
		${WRKSRC}/wmbiff/sample.wmbiffrc

post-install:
>>>>>>> upstream/main
	${INSTALL_DATA} ${WRKSRC}/wmbiff/sample.wmbiffrc ${STAGEDIR}${DATADIR}

.include <bsd.port.mk>
