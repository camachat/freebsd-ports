PORTNAME=	bsdisks
DISTVERSION=	0.35
CATEGORIES=	sysutils
PKGNAMESUFFIX=	-${FLAVOR}

MAINTAINER=	arrowd@FreeBSD.org
COMMENT=	UDisks2 service implementation for FreeBSD
WWW=		https://foss.heptapod.net/bsdutils/bsdisks

LICENSE=	BSD3CLAUSE

FLAVORS=	qt5 qt6
FLAVOR?=	qt5

LIB_DEPENDS=	libpolkit-qt${FLAVOR:C/qt//}-core-1.so:sysutils/polkit-qt-1@${FLAVOR}

USES=		cmake compiler:c++11-lang qt:${FLAVOR:C/qt//}
USE_QT=	${_USE_QT_${FLAVOR}}
CMAKE_ARGS=	-DQT_MAJOR_VERSION=${FLAVOR:C/qt//}
_USE_QT_qt5=		qmake:build buildtools:build core dbus
_USE_QT_qt6=		base

USE_GITLAB=		yes
GL_SITE=		https://foss.heptapod.net
GL_ACCOUNT=		bsdutils
GL_TAGNAME=		205d242c512099074c3c1bf939a4fd83139bb5c8

.include <bsd.port.mk>
