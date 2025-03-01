/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmqt_config.h"
#include <QApplication>
#include <QLabel>
#include <QtGui>

#include "pqt_database.h"
#include "pqt_findscu.h"
#include "pqt_main_window.h"

static void
initialize_application (void)
{
    /* Set path to persistent application settings */
    QCoreApplication::setOrganizationName ("Plastimatch");
    QCoreApplication::setOrganizationDomain ("plastimatch.org");
    QCoreApplication::setApplicationName ("plastimatch_qt");

    /* QT doesn't seem to have an API for getting the user's application 
       data directory.  So we construct a hypothetical ini file name, 
       then grab the directory. */
    QSettings tmp (
	QSettings::IniFormat, /* Make sure we get path, not registry */
	QSettings::UserScope, /* Get user directory, not system direcory */
	"Plastimatch",        /* Orginazation name (subfolder within path) */
	"plastimatch_qt"      /* Application name (file name with subfolder) */
    );
    QString config_dir = QFileInfo(tmp.fileName()).absolutePath();

#if defined (commentout)
    QMessageBox::information (0, QString ("Info"), 
	QString ("Config dir is %1").arg (config_dir));
#endif

    /* Construct filename of sqlite database that holds settings.
       On unix, this is $HOME/.config/Plastimatch/pqt.sqlite
    */
    QSettings settings;
    QString db_path = settings.value ("db/sqlite3_path", 
	QFileInfo (QDir (config_dir), QString ("pqt.sqlite"))
	.absoluteFilePath()).toString();

    /* Load database */
    printf ("Starting database\n");
    pqt_database_start (db_path);
}

int
main (int argc, char **argv)
{
    int rc;
    QApplication app (argc, argv);

    initialize_application ();

    Pqt_main_window pqt_main_window;
    pqt_main_window.show ();

    rc = app.exec();

    pqt_database_stop ();
    
    /* Application emits database warning on program exit.  Apparently 
       this is a bug in Qt.
       Ref: http://lists.trolltech.com/qt-interest/2008-05/msg00553.html */
    return rc;
}
