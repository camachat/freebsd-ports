# Created by: Slawomir Wojciech Wojtczak <vermaden@interia.pl>
# $FreeBSD$

PORTNAME=	automount
PORTVERSION=	1.7.0
CATEGORIES=	sysutils
MASTER_SITES=	https://raw.github.com/vermaden/automount/master/ \
		LOCAL/madpilot/${PORTNAME}/

MAINTAINER=	vermaden@interia.pl
COMMENT=	FreeBSD's devd(8) based automount solution

LICENSE=	BSD2CLAUSE

PLIST_FILES=	sbin/automount etc/devd/automount_devd.conf etc/automount.conf.sample

SUB_FILES=	pkg-install
NO_BUILD=	yes
NO_ARCH=	yes

OPTIONS_DEFINE=		NTFS3G EXT4 EXFAT HFS XFS MTP
OPTIONS_DEFAULT=	NTFS3G EXT4 EXFAT MTP
NTFS3G_DESC=		Enable NTFS write support with ntfs-3g over FUSE
NTFS3G_RUN_DEPENDS=	fusefs-ntfs>=0:sysutils/fusefs-ntfs
EXT4_DESC=		Support EXT4 filesystem
EXT4_RUN_DEPENDS=	fusefs-ext4fuse>=0:sysutils/fusefs-ext4fuse
EXFAT_DESC=		Support Microsoft exFAT filesystem
EXFAT_RUN_DEPENDS=	fusefs-exfat>=0:sysutils/fusefs-exfat
HFS_DESC=		Support HFS+ filesystem
HFS_RUN_DEPENDS=	fusefs-hfsfuse>=0:sysutils/fusefs-hfsfuse
XFS_DESC=		Support XFS filesystem
XFS_RUN_DEPENDS=	fusefs-lkl>=0:sysutils/fusefs-lkl
MTP_DESC=		Support MTP filesystem
MTP_RUN_DEPENDS=	fusefs-simple-mtpfs>=0:sysutils/fusefs-simple-mtpfs \
                    zenity>=0:x11/zenity

do-install:
	${INSTALL_SCRIPT} ${WRKDIR}/automount             ${STAGEDIR}${PREFIX}/sbin
	${INSTALL_DATA}   ${WRKDIR}/automount_devd.conf   ${STAGEDIR}${PREFIX}/etc/devd/automount_devd.conf
	${INSTALL_DATA}   ${WRKDIR}/automount.conf	  ${STAGEDIR}${PREFIX}/etc/automount.conf.sample

.include <bsd.port.mk>

