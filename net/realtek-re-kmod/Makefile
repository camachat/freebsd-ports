PORTNAME=	re
PORTVERSION=	1100.00
DISTVERSIONPREFIX=	v
PORTREVISION=	1
CATEGORIES=	net
PKGNAMEPREFIX=	realtek-
PKGNAMESUFFIX=	-kmod

MAINTAINER=	ale@FreeBSD.org
COMMENT=	Kernel driver for Realtek PCIe Ethernet Controllers
WWW=		https://github.com/alexdupre/rtl_bsd_drv

LICENSE=	BSD4CLAUSE

USES=		kmod tar:tgz uidfix

USE_GITHUB=	yes
GH_ACCOUNT=	alexdupre
GH_PROJECT=	rtl_bsd_drv
GH_TAGNAME=	ea4ed1e

.include <bsd.port.options.mk>

.if ${OPSYS} == FreeBSD && (${OSVERSION} >= 1500000)
EXTRA_PATCHES+= ${FILESDIR}/extra-patch-if__re.c
.endif

PLIST_FILES=	${KMODDIR}/if_re.ko

.include <bsd.port.mk>
