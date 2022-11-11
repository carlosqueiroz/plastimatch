/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <iostream>
#include <QtGlobal>
#include <QApplication>
#include <QDebug>
#include <QLabel>
#if QT_VERSION >= QT_VERSION_CHECK (5,0,0)
#include <QSslSocket>
#endif

int
main (int argc, char **argv)
{
    // Display designer path
    QStringList paths = QCoreApplication::libraryPaths();
    for (QStringList::iterator it = paths.begin(); it!=paths.end(); it++) {
        std::cout << "Looking for plugins at path: "
            << it->toStdString() << std::endl;
    }

    QApplication app(argc, argv);
    QLabel *label = new QLabel("Hello World!");
  
    label->show();

    qDebug() << "Qt version is " << qVersion();

#if QT_VERSION >= QT_VERSION_CHECK (5,4,0)
    qDebug() << "Qt was built with SSL "
        << QSslSocket::sslLibraryBuildVersionString();
#endif
#if QT_VERSION >= QT_VERSION_CHECK (5,0,0)
    qDebug() << "Qt is linked with SSL "
        << QSslSocket::sslLibraryVersionString();
#endif
    
    return app.exec();
}
