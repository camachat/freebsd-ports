PORTNAME=	pyradio
DISTVERSION=	0.9.3.11.9
PORTREVISION=	0
CATEGORIES=	audio python
PKGNAMEPREFIX=	${PYTHON_PKGNAMEPREFIX}

MAINTAINER=	nivit@FreeBSD.org
COMMENT=	Curses based Internet radio player
WWW=		https://github.com/coderholic/pyradio

LICENSE=	MIT
LICENSE_FILE=	${WRKSRC}/LICENSE

BUILD_DEPENDS=	${PKGNAMEPREFIX}dnspython>=2.2.1:dns/py-dnspython@${PY_FLAVOR} \
		${PKGNAMEPREFIX}netifaces>=0.11.0:net/py-netifaces@${PY_FLAVOR} \
		${PKGNAMEPREFIX}psutil>=5.9.2:sysutils/py-psutil@${PY_FLAVOR} \
		${PKGNAMEPREFIX}python-dateutil>=2.8.2:devel/py-python-dateutil@${PY_FLAVOR} \
		${PKGNAMEPREFIX}requests>=2.28.1:www/py-requests@${PY_FLAVOR} \
		${PKGNAMEPREFIX}setuptools>=63.1.0:devel/py-setuptools@${PY_FLAVOR} \
		${PKGNAMEPREFIX}wheel>=0.38.4:devel/py-wheel@${PY_FLAVOR} \
		bash:shells/bash
RUN_DEPENDS=	${PKGNAMEPREFIX}dnspython>=2.2.1:dns/py-dnspython@${PY_FLAVOR} \
		${PKGNAMEPREFIX}netifaces>=0.11.0:net/py-netifaces@${PY_FLAVOR} \
		${PKGNAMEPREFIX}psutil>=5.9.2:sysutils/py-psutil@${PY_FLAVOR} \
		${PKGNAMEPREFIX}python-dateutil>=2.8.2:devel/py-python-dateutil@${PY_FLAVOR} \
		${PKGNAMEPREFIX}requests>=2.28.1:www/py-requests@${PY_FLAVOR} \
		${PKGNAMEPREFIX}rich>=13.3.2:textproc/py-rich@${PY_FLAVOR} \
		${PKGNAMEPREFIX}wheel>=0.38.4:devel/py-wheel@${PY_FLAVOR} \
		bash:shells/bash \
		xdg-open:devel/xdg-utils

USES=		python
USE_GITHUB=	yes
GH_ACCOUNT=	coderholic
USE_PYTHON=	autoplist pep517

NO_ARCH=	yes

PLIST_FILES=	share/applications/${PORTNAME}.desktop \
		share/pixmaps/${PORTNAME}.png
PORTDOCS=	*

OPTIONS_DEFINE=			DOCS METADATA
OPTIONS_DEFAULT=		METADATA MPV
OPTIONS_SINGLE=			MEDIA_PLAYER
OPTIONS_SINGLE_MEDIA_PLAYER=	MPLAYER MPV VLC

METADATA_DESC=	Insert tags, chapters and cover into recordings
# override predefined *_DESC
MPV_DESC=	Mpv media player support
VLC_DESC=	VLC media player support

DOCS_PLIST_FILES=	share/man/man1/pyradio-client.1.gz \
			share/man/man1/pyradio.1.gz \
			share/man/man1/pyradio_buf.1.gz \
			share/man/man1/pyradio_rb.1.gz \
			share/man/man1/pyradio_rec.1.gz \
			share/man/man1/pyradio_server.1.gz
METADATA_RUN_DEPENDS+=	mkvinfo:multimedia/mkvtoolnix@nogui
MPLAYER_RUN_DEPENDS+=	mplayer:multimedia/mplayer
MPV_RUN_DEPENDS+=	mpv:multimedia/mpv
VLC_RUN_DEPENDS+=	vlc:multimedia/vlc

post-patch:
	@${SED} -i.bak -e 's/\(distro =\) None/\1 ${OPSYS}/' \
	-e 's/\(enable_notifications =\) -1/\1 0/' \
	-e 's/\(remove_station_icons  =\) True/\1 False/' \
	-e 's/\(xdg_compliant =\) False/\1 True/' \
	 	${WRKSRC}/pyradio/config

post-install:
	${INSTALL_DATA} ${WRKSRC}/devel/${PORTNAME}.desktop \
		${STAGEDIR}${DESKTOPDIR}/${PORTNAME}.desktop
	${INSTALL_DATA} ${WRKSRC}/devel/${PORTNAME}.png \
		${STAGEDIR}${PREFIX}/share/pixmaps/${PORTNAME}.png

post-install-DOCS-on:
	@${MKDIR} ${STAGEDIR}${DOCSDIR}
	${INSTALL_MAN} ${WRKSRC}/docs/pyradio*.1 ${STAGEDIR}${PREFIX}/share/man/man1
	${INSTALL_DATA} ${WRKSRC}/docs/*.html ${WRKSRC}/*.md ${STAGEDIR}${DOCSDIR}/

.include <bsd.port.mk>
