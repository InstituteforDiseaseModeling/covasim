class FullTimeContactManager():
    ''' Contact manager for regular 5-day-per-week school '''

    def __init__(self, uids, contacts, is_hybrid, sim):
        self.contacts = contacts
        self.removed_contacts = {} # Dictionary uid: contacts | group?

        #cvb.Layer

    def remove_individual(self, uids):
        ''' Remove one or more individual from the contact network '''

    def restore_individual(self, uids):
        ''' Restore one or more individual to the contact network '''

    def get_contacts(self, group):


class School():

    def __init__(self, school_id, school_type, uids, layer, is_hybrid, sim):
        self.school_id = school_iud
        self.school_type = school_type
        self.uids = uids
        self.is_hybrid = is_hybrid
        self.sim = sim

        self.is_open = False # Schools start closed

        self.sick_uids = [] # ? list per day to be able to easily return to school? Timer?

        if self.is_hybrid:
            self.schedule = {
                'Monday':    'A',
                'Tuesday':   'A',
                'Wednesday': 'distance',
                'Thursday':  'B',
                'Friday':    'B',
                'Saturday':  'no_school',
                'Sunday':    'no_school',
                }
        else:
            self.schedule = {
                'Monday':    'all',
                'Tuesday':   'all',
                'Wednesday': 'all',
                'Thursday':  'all',
                'Friday':    'all',
                'Saturday':  'no_school',
                'Sunday':    'no_school',
            }



    def close(self):
        self.is_open = False

    def open(self):
        self.is_open = True

    def screen_test_trace(self, screen_frac=1, test_frac=0.5, trace_frac=0.5):
        # Only screen those AT school

    def test_undiagnosed(self, student_frac=0, staff_frac=0, teacher_frac=0):
        # Administer diagnostic tests in individuals not already diagnosed
        # Exclude those home sick?

    def _contact_trace(self, inds):

    def update(self, t):
        # Process the day, return in school contacts

        # First check if school is open
        if not school.is_open:
            return None

        date = sim.date(t)
        dayname = sc.readdate(date).strftime('%A')
        group = mapping[dayname]

        # Perform symptom screening
        self.screen_test_trace()
