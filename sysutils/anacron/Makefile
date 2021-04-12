# Created by: Derik van Zuetphen <dz@426.ch>
<<<<<<< HEAD
# $FreeBSD$

PORTNAME=	anacron
PORTVERSION=	2.3
PORTREVISION=	6
=======

PORTNAME=	anacron
PORTVERSION=	2.3
PORTREVISION=	7
>>>>>>> upstream/main
CATEGORIES=	sysutils
MASTER_SITES=	SF

MAINTAINER=	dz@426.ch
COMMENT=	Schedules periodic jobs on systems that are not permanently up

LICENSE=	GPLv2+
LICENSE_FILE=	${WRKSRC}/COPYING

USES=		gmake
<<<<<<< HEAD
USE_RC_SUBR=	anacron
SUB_FILES=	pkg-message

post-patch:
.for f in Makefile anacron.8 anacrontab.5
	@${REINPLACE_CMD} -e 's|\(/etc/anacrontab\)|${PREFIX}\1|' ${WRKSRC}/${f}
=======
USE_RC_SUBR=	${PORTNAME}
SUB_FILES=	pkg-message
PLIST_SUB=	PORTNAME=${PORTNAME}
SUB_LIST+=	PORTNAME=${PORTNAME} COMMENT="${COMMENT}"

post-patch:
.for f in Makefile ${PORTNAME}.8 ${PORTNAME}tab.5
	@${REINPLACE_CMD} -e 's|\(/etc/${PORTNAME}tab\)|${PREFIX}\1|' ${WRKSRC}/${f}
>>>>>>> upstream/main
.endfor
	@${REINPLACE_CMD} -e 's|^CFLAGS = .*||' ${WRKSRC}/Makefile

do-install:
<<<<<<< HEAD
	${INSTALL_PROGRAM} ${WRKSRC}/anacron ${STAGEDIR}${PREFIX}/sbin
	${INSTALL_MAN} ${WRKSRC}/anacron.8 ${STAGEDIR}${MAN8PREFIX}/man/man8
	${INSTALL_MAN} ${WRKSRC}/anacrontab.5 ${STAGEDIR}${MAN5PREFIX}/man/man5
	${INSTALL_DATA} ${FILESDIR}/anacrontab.sample ${STAGEDIR}${PREFIX}/etc
	@${MKDIR} ${STAGEDIR}/var/spool/anacron
=======
	${INSTALL_PROGRAM} ${WRKSRC}/${PORTNAME} ${STAGEDIR}${PREFIX}/sbin
	${INSTALL_MAN} ${WRKSRC}/${PORTNAME}.8 ${STAGEDIR}${MAN8PREFIX}/man/man8
	${INSTALL_MAN} ${WRKSRC}/${PORTNAME}tab.5 ${STAGEDIR}${MAN5PREFIX}/man/man5
	${INSTALL_DATA} ${FILESDIR}/${PORTNAME}tab.sample ${STAGEDIR}${PREFIX}/etc
	@${MKDIR} ${STAGEDIR}/var/spool/${PORTNAME}
>>>>>>> upstream/main

.include <bsd.port.mk>
