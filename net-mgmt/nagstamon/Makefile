# Created by: Emanuel Haupt <ehaupt@FreeBSD.org>
<<<<<<< HEAD
# $FreeBSD$

PORTNAME=	nagstamon
PORTVERSION=	3.4.1
PORTREVISION=	6
CATEGORIES=	net-mgmt python
MASTER_SITES=	https://nagstamon.ifw-dresden.de/files/stable/ \
		LOCAL/ehaupt
DISTNAME=	Nagstamon-${PORTVERSION}
=======

PORTNAME=	nagstamon
PORTVERSION=	3.6.0
DISTVERSIONPREFIX=	v
PORTREVISION=	1
CATEGORIES=	net-mgmt python
>>>>>>> upstream/main

MAINTAINER=	ehaupt@FreeBSD.org
COMMENT=	Nagios status monitor for your desktop

LICENSE=	GPLv2
LICENSE_FILE=	${WRKSRC}/LICENSE

BUILD_DEPENDS=	${PYTHON_PKGNAMEPREFIX}keyring>0:security/py-keyring@${PY_FLAVOR} \
		${PYTHON_PKGNAMEPREFIX}psutil>=2.0:sysutils/py-psutil@${PY_FLAVOR}
RUN_DEPENDS=	${PYTHON_PKGNAMEPREFIX}beautifulsoup>=4.2.1:www/py-beautifulsoup@${PY_FLAVOR} \
		${PYTHON_PKGNAMEPREFIX}keyring>0:security/py-keyring@${PY_FLAVOR} \
		${PYTHON_PKGNAMEPREFIX}psutil>0:sysutils/py-psutil@${PY_FLAVOR} \
		${PYTHON_PKGNAMEPREFIX}requests-kerberos>0:security/py-requests-kerberos@${PY_FLAVOR}

USES=		cpe pyqt:5 python:3.6+
CPE_VENDOR=	henri_wahl
<<<<<<< HEAD
=======
USE_GITHUB=	yes
GH_ACCOUNT=	HenriWahl
GH_PROJECT=	Nagstamon
GH_TAGNAME=	latest
>>>>>>> upstream/main
USE_PYQT=	core_run dbus_run gui_run multimedia_run network_run sip_run \
		svg_run widgets_run
USE_PYTHON=	autoplist distutils noflavors

NO_ARCH=	yes

<<<<<<< HEAD
WRKSRC=		${WRKDIR}/Nagstamon

=======
>>>>>>> upstream/main
.include <bsd.port.mk>
