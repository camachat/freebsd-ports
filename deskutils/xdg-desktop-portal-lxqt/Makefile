PORTNAME=	xdg-desktop-portal-lxqt
PORTVERSION=	1.2.0
PORTREVISION=	1
CATEGORIES=	deskutils

MAINTAINER=	lxqt@FreeBSD.org
COMMENT=	Portal frontend service for LXQT
WWW=		https://github.com/lxqt/xdg-desktop-portal-lxqt

LICENSE=	LGPL21

USES=		cmake kde:6 lxqt:2 pkgconfig qt:6 tar:xz
USE_KDE=	windowsystem
USE_LXQT=	libfmqt6
USE_QT=		base:run tools:build

post-install:
	@${RM} -r ${STAGEDIR}${PREFIX}/lib/systemd

.include <bsd.port.mk>
