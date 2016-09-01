"""
Unit test of the streams system
"""
import param
from holoviews.element.comparison import ComparisonTestCase
from holoviews.streams import Stream, PositionX, PositionY, PositionXY, ParamValues
from holoviews.streams import Rename, Group


class TestSubscriber(object):

    def __init__(self):
        self.call_count = 0
        self.kwargs = None

    def __call__(self, **kwargs):
        self.call_count += 1
        self.kwargs = kwargs


class TestPositionStreams(ComparisonTestCase):

    def test_positionX_init(self):
        PositionX()

    def test_positionXY_init_values(self):
        position = PositionXY(x=1, y=3)
        self.assertEqual(position.value, dict(x=1, y=3))

    def test_positionXY_update_values(self):
        position = PositionXY()
        position.update(x=5, y=10)
        self.assertEqual(position.value, dict(x=5, y=10))

    def test_positionY_const_parameter(self):
        position = PositionY()
        try:
            position.y = 5
            raise Exception('No constant parameter exception')
        except TypeError as e:
            self.assertEqual(str(e), "Constant parameter 'y' cannot be modified")


class TestParamValuesStream(ComparisonTestCase):

    def setUp(self):

        class Inner(param.Parameterized):

            x = param.Number(default = 0)
            y = param.Number(default = 0)

        self.inner = Inner

    def tearDown(self):
        self.inner.x = 0
        self.inner.y = 0

    def test_object_value(self):
        obj = self.inner()
        stream = ParamValues(obj)
        self.assertEqual(stream.value, {'x':0, 'y':0})

    def test_class_value(self):
        stream = ParamValues(self.inner)
        self.assertEqual(stream.value, {'x':0, 'y':0})

    def test_object_value_update(self):
        obj = self.inner()
        stream = ParamValues(obj)
        self.assertEqual(stream.value, {'x':0, 'y':0})
        stream.update(x=5, y=10)
        self.assertEqual(stream.value, {'x':5, 'y':10})

    def test_class_value_update(self):
        stream = ParamValues(self.inner)
        self.assertEqual(stream.value, {'x':0, 'y':0})
        stream.update(x=5, y=10)
        self.assertEqual(stream.value, {'x':5, 'y':10})



class TestSubscribers(ComparisonTestCase):

    def test_exception_subscriber(self):
        subscriber = TestSubscriber()
        position = PositionXY(subscribers=[subscriber])
        kwargs = dict(x=3, y=4)
        position.update(**kwargs)
        self.assertEqual(subscriber.kwargs, kwargs)

    def test_subscriber_disabled(self):
        subscriber = TestSubscriber()
        position = PositionXY(subscribers=[subscriber])
        kwargs = dict(x=3, y=4)
        position.update(trigger=False, **kwargs)
        self.assertEqual(subscriber.kwargs, None)


    def test_subscribers(self):
        subscriber1 = TestSubscriber()
        subscriber2 = TestSubscriber()
        position = PositionXY(subscribers=[subscriber1, subscriber2])
        kwargs = dict(x=3, y=4)
        position.update(**kwargs)
        self.assertEqual(subscriber1.kwargs, kwargs)
        self.assertEqual(subscriber2.kwargs, kwargs)

    def test_batch_subscriber(self):
        subscriber = TestSubscriber()

        positionX = PositionX(subscribers=[subscriber])
        positionY = PositionY(subscribers=[subscriber])

        positionX.update(trigger=False, x=5)
        positionY.update(trigger=False, y=10)

        Stream.trigger([positionX, positionY])
        self.assertEqual(subscriber.kwargs, dict(x=5, y=10))
        self.assertEqual(subscriber.call_count, 1)

    def test_batch_subscribers(self):
        subscriber1 = TestSubscriber()
        subscriber2 = TestSubscriber()

        positionX = PositionX(subscribers=[subscriber1, subscriber2])
        positionY = PositionY(subscribers=[subscriber1, subscriber2])

        positionX.update(trigger=False, x=50)
        positionY.update(trigger=False, y=100)

        Stream.trigger([positionX, positionY])

        self.assertEqual(subscriber1.kwargs, dict(x=50, y=100))
        self.assertEqual(subscriber1.call_count, 1)

        self.assertEqual(subscriber2.kwargs, dict(x=50, y=100))
        self.assertEqual(subscriber2.call_count, 1)


class TestPreprocessors(ComparisonTestCase):

    def test_rename_preprocessor(self):
        position = PositionXY([Rename(x='x1',y='y1')], x=1, y=3)
        self.assertEqual(position.value, dict(x1=1, y1=3))

    def test_group_preprocessor(self):
        position = PositionXY([Group('mygroup')], x=1, y=3)
        self.assertEqual(position.value, dict(mygroup={'x':1,'y':3}))
