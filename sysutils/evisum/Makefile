PORTNAME=	evisum
PORTVERSION=	0.6.4
CATEGORIES=	sysutils enlightenment
MASTER_SITES=	http://download.enlightenment.org/rel/apps/${PORTNAME}/

MAINTAINER=	enlightenment@FreeBSD.org
COMMENT=	EFL process monitor and system monitor
WWW=		https://www.enlightenment.org/

LICENSE=	BSD2CLAUSE

LIB_DEPENDS=	libefl.so:devel/efl

DIST_SUBDIR=	enlightenment
USES=		compiler:c11 desktop-file-utils meson pkgconfig tar:xz

PLIST_FILES=	"@(root,kmem,2755) bin/evisum"

.include <bsd.port.mk>
