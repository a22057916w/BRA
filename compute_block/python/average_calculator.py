import queue
class ACBlock:
    def __init__(self, total_step, period_step):
        if total_step % period_step != 0:
            raise ValueError('parameter total_step must be divisible by paremeter period_step.')
        self.AC_MAX_LEN = round(total_step / period_step)
        self.AC_END = round(period_step)
        self.record = queue.Queue(maxsize=self.AC_MAX_LEN)
        self.zero_buffer = {'numerator':0, 'denominator':0}
        self.period_buffer = {'numerator':0, 'denominator':0}
        self.output_buffer = 0.0
        self.counter = 1

    def step(self, numerator, denominator):
        self.zero_buffer['numerator'] += numerator
        self.zero_buffer['denominator'] += denominator

        if self.counter != self.AC_END:
            self.counter += 1
            return self.output_buffer, False
        
        self.counter = 1

        self.record.put(self.zero_buffer)

        self.period_buffer['numerator'] += self.zero_buffer['numerator']
        self.period_buffer['denominator'] += self.zero_buffer['denominator']

        self.zero_buffer = {'numerator':0, 'denominator':0}

        if self.period_buffer['denominator'] != 0:
            self.output_buffer = self.period_buffer['numerator'] / self.period_buffer['denominator']
        else:
            self.output_buffer = 0.0

        if self.record.full():
            tmp = self.record.get()
            self.period_buffer['numerator'] -= tmp['numerator']
            self.period_buffer['denominator'] -= tmp['denominator']

        return self.output_buffer, True