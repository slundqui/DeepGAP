class Parent(object):
    def __init__(self):
        print "Parent Constructor"

class Child(Parent):
    def __init__(self, newParam):
        super(Child, self).__init__()
        print "Child Constructor"




c = Child("asdf")
