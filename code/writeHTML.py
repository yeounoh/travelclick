#!/usr/local/bin/python
def writeHTML(figs):
    with open('index.html','w') as f:
        f.write('<script type="text/javascript" src="http://d3js.org/d3.v3.min.js"></script>\n')
        f.write('<script type="text/javascript" src="http://mpld3.github.io/js/mpld3.v0.2.js"></script>\n')
        f.write('''<style>
                .small {
                    background-color:white;
                    width:200px;
                    max-width:100%;
                    float:left;
                    padding:10px;
                }
                .border {
                    boackground-color:black;
                    clear:both;
                    padding:5px;
                }

                .dropbtn {
                    background-color: #4CAF50;
                    color: white;
                    padding: 16px;
                    font-size: 16px;
                    border: none;
                    cursor: pointer;
                }

                /* Dropdown button on hover & focus */
                .dropbtn:hover, .dropbtn:focus {
                    background-color: #3e8e41;
                }

                /* The container <div> - needed to position the dropdown content */
                .dropdown {
                    position: relative;
                    display: inline-block;
                }

                /* Dropdown Content (Hidden by Default) */
                .dropdown-content {
                    display: none;
                    position: absolute;
                    background-color: #f9f9f9;
                    min-width: 160px;
                    box-shadow: 0px 8px 16px 0px rgba(0,0,0,0.2);
                }
                /* Links inside the dropdown */
                .dropdown-content a {
                    color: black;
                    padding: 12px 16px;
                    text-decoration: none;
                    display: block;
                }

                /* Change color of dropdown links on hover */
                .dropdown-content a:hover {background-color: #f1f1f1}

                .show {display:block;}
                </style>\n''')
        f.write('''
                <div class="dropdown">
                    <button onclick="myFunction()" class="dropbtn">Dropdown</button>
                    <div id="myDropdown" class="dropdown-content">
                    <a href="#">Link 1</a>
                    <a href="#">Link 2</a>
                    <a href="#">Link 3</a>
                    </div>
                </div>
                \n''')
        for fig in figs:
            f.write('<div class="small" id=\"fig' + str(figs.index(fig)) + '\"></div>\n')
            f.write('<div class="border"></div>\n')
        f.write('<script type=\"text/javascript\">\n')
        for fig in figs:
            f.write('   var json' + str(figs.index(fig)) + ' =  ' + fig + ' ;\n')
        for fig in figs:
            f.write('   mpld3.draw_figure(\"fig' + str(figs.index(fig)) +'\", json' + str(figs.index(fig)) + ');\n')
        f.write('''
                function myFunction() {
                    document.getElementById("myDropdown").classList.toggle("show");
                }
                window.onclick = function(event) {
                    if (!event.target.matches('.dropbtn')) {
                        var dropdowns = document.getElementsByClassName("dropdown-content");
                        var i;
                        for (i = 0; i < dropdowns.length; i++) {
                            var openDropdown = dropdowns[i];
                            if (openDropdown.classlist.contains('show')) {
                                openDropdown.classList.remove('show');
                            }
                        }
                    }
                }
                \n''')
        f.write('</script>\n')

