/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _parameter_parser_h_
#define _parameter_parser_h_

#include "plmbase_config.h"
#include "plm_return_code.h"

/*! \brief 
 * The Parameter_parser class is an abstract base class which is used 
 * to parse ini-style file formats that control the registration, mabs, 
 * and dose calculation codes.
 */
class PLMBASE_API Parameter_parser {
public:
    Parameter_parser ();
public:
    bool key_regularization;
    bool empty_values_allowed;
    std::string default_index;
public:
    /* Callbacks */
    virtual Plm_return_code begin_section (
        const std::string& section) = 0;
    virtual Plm_return_code end_section (
        const std::string& section) = 0;
    virtual Plm_return_code set_key_value (
        const std::string& section, 
        const std::string& key, 
        const std::string& index, 
        const std::string& member, 
        const std::string& val) = 0;

    /*! \brief Key regularization convert the key field to lowercase and 
      changes hyphens to underscores.  Only the key is modified. 
      Pass in "true" to enable key regularization, or "false" to 
      disable it.   Default is "true". */
    void enable_key_regularization (
        bool enable
    );

    /*! \brief If a key does not have an equal sign, it will be 
      considered a syntax error unless empty values are allowed.
    */
    void allow_empty_values (
        bool enable
    );

    /*! \brief Choose what index is passed to set_key_value() 
      when no index is found in the file.  Default is "". */
    void set_default_index (std::string& default_index);
    void set_default_index (const char *default_index);

    /* Return zero if config string is correctly parsed */
    Plm_return_code parse_config_string (
        const char* config_string
    );
    Plm_return_code parse_config_string (
        const std::string& config_string
    );
    Plm_return_code parse_config_file (
        const char* config_fn
    );
    Plm_return_code parse_config_file (
        const std::string& config_fn
    );
};

#endif
