<<<<<<< HEAD
# $FreeBSD$

PORTNAME=	ustreamer
DISTVERSIONPREFIX=	v
DISTVERSION=	3.23
=======
PORTNAME=	ustreamer
DISTVERSIONPREFIX=	v
DISTVERSION=	3.25
>>>>>>> upstream/main
CATEGORIES=	multimedia net

MAINTAINER=	yuri@FreeBSD.org
COMMENT=	Lightweight and fast MJPG-HTTP streamer

LICENSE=	GPLv3
LICENSE_FILE=	${WRKSRC}/LICENSE

BUILD_DEPENDS=	${LOCALBASE}/include/linux/videodev2.h:multimedia/v4l_compat
LIB_DEPENDS=	libevent.so:devel/libevent \
		libuuid.so:misc/e2fsprogs-libuuid

USES=		compiler:c11 gmake jpeg localbase:ldflags
USE_GITHUB=	yes
GH_ACCOUNT=	pikvm

<<<<<<< HEAD
CFLAGS+=	-DWITH_PTHREAD_NP=0 # src/threading.h:101:4: error: thread_get_name() not implemented, you can disable it using WITH_PTHREAD_NP=0
=======
MAKE_ARGS=	WITH_PTHREAD_NP=0
>>>>>>> upstream/main

INSTALL_TARGET=	install-strip

PLIST_FILES=	bin/ustreamer \
		bin/ustreamer-dump \
		man1/ustreamer-dump.1.gz \
		man1/ustreamer.1.gz

<<<<<<< HEAD
BROKEN_FreeBSD_11=	error: thread_get_name() not implemented, you can disable it using WITH_PTHREAD_NP=0
=======
.include <bsd.port.options.mk>

# strerror_l appeared in 12 in r368915 on 2021-01-01, OSVERSION was last changed to 1202505 in r368887 on 2020-12-29
post-patch:
.if ${OPSYS} == "FreeBSD" && ${OSVERSION} <= 1202505
	@${REINPLACE_CMD} -e 's|strerror_l(error, locale)|strerror(error)|' ${WRKSRC}/src/libs/tools.h
.endif
>>>>>>> upstream/main

.include <bsd.port.mk>
