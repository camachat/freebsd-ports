# Created by: Bernhard Froehlich <decke@bluelife.at>
<<<<<<< HEAD
# $FreeBSD$

PORTNAME=	phpvirtualbox
DISTVERSION=	5.2-1
PORTREVISION=	1
=======

PORTNAME=	phpvirtualbox
DISTVERSION=	6.1
>>>>>>> upstream/main
CATEGORIES=	www

MAINTAINER=	vbox@FreeBSD.org
COMMENT=	AJAX Web Interface for VirtualBox

LICENSE=	GPLv3

USES=		dos2unix php
USE_PHP=	json session simplexml soap xml

NO_ARCH=	yes
NO_BUILD=	yes

USE_GITHUB=	yes
<<<<<<< HEAD
=======
GH_TAGNAME=	cf82d58f2
>>>>>>> upstream/main

DOS2UNIX_REGEX=	.*\.(php|txt|js|css|html)
ETCDIR=		${WWWDIR}

post-patch:
<<<<<<< HEAD
	@${REINPLACE_CMD} -e 's#/usr/share/phpvirtualbox#${WWWDIR}#g' ${WRKSRC}/phpvirtualbox.conf
=======
	@${REINPLACE_CMD} -e 's#/usr/share/phpvirtualbox#${WWWDIR}#g' \
		${WRKSRC}/phpvirtualbox.conf
>>>>>>> upstream/main

do-install:
	${MKDIR} ${STAGEDIR}${WWWDIR}
	(cd ${WRKSRC}/ && ${COPYTREE_SHARE} "*" ${STAGEDIR}${WWWDIR} \
	    "-not -name *.orig -not -name config.php-example")
	${INSTALL_DATA} ${WRKSRC}/config.php-example \
	    ${STAGEDIR}${WWWDIR}/config.php.sample

.include <bsd.port.mk>
