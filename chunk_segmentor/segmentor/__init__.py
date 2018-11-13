import os
import sys
from jpype import JClass, startJVM, shutdownJVM, isJVMStarted
from jpype import getDefaultJVMPath, isThreadAttachedToJVM, attachThreadToJVM

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(curdir, os.path.pardir))


if sys.version_info[0] < 3:
    print('Please use Python 3')


'''
Load global variables
'''
STATIC_ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'static')
HANLP_JAR_VERSION = '1.6.7'
HANLP_JAR_PATH = os.path.join(STATIC_ROOT, 'hanlp-test-{}.jar'.format(HANLP_JAR_VERSION))

HANLP_JVM_XMS = "1g"
HANLP_JVM_XMX = "2g"
if os.path.exists(HANLP_JAR_PATH) and os.path.exists(STATIC_ROOT):
    PATH_CONFIG = os.path.join(STATIC_ROOT, 'hanlp.properties')
    # HANLP_JAR_VERSION = os.path.basename(HANLP_JAR_PATH)[len('hanlp-'):-len('.jar')]
    HANLP_DATA_PATH = os.path.join(STATIC_ROOT, 'data')
else:
    raise BaseException(
        "Error: %s or %s does not exists." %
        (HANLP_JAR_PATH, STATIC_ROOT))

JAVA_JAR_CLASSPATH = "-Djava.class.path=%s%s%s" % (
    HANLP_JAR_PATH, os.pathsep, STATIC_ROOT)


# 启动JVM
startJVM(
    getDefaultJVMPath(),
    JAVA_JAR_CLASSPATH,
    "-Xms%s" %
    HANLP_JVM_XMS,
    "-Xmx%s" %
    HANLP_JVM_XMX)


def _attach_jvm_to_thread():
    """
    use attachThreadToJVM to fix multi-thread issues: https://github.com/hankcs/pyhanlp/issues/7
    """
    if not isThreadAttachedToJVM():
        attachThreadToJVM()


class SafeJClass(object):
    def __init__(self, proxy):
        """
        JClass的线程安全版
        :param proxy: Java类的完整路径，或者一个Java对象
        """
        self._proxy = JClass(proxy) if type(proxy) is str else proxy

    def __getattr__(self, attr):
        _attach_jvm_to_thread()
        return getattr(self._proxy, attr)

    def __call__(self, *args):
        if args:
            proxy = self._proxy(*args)
        else:
            proxy = self._proxy()
        return SafeJClass(proxy)


class LazyLoadingJClass(object):
    def __init__(self, proxy):
        """
        惰性加载Class。仅在实际发生调用时才触发加载，适用于包含资源文件的静态class
        :param proxy:
        """
        self._proxy = proxy

    def __getattr__(self, attr):
        _attach_jvm_to_thread()
        self._lazy_load_jclass()
        return getattr(self._proxy, attr)

    def _lazy_load_jclass(self):
        if type(self._proxy) is str:
            self._proxy = JClass(self._proxy)

    def __call__(self, *args):
        self._lazy_load_jclass()
        if args:
            proxy = self._proxy(*args)
        else:
            proxy = self._proxy()
        return SafeJClass(proxy)


# API列表
'''
CustomDictionary = LazyLoadingJClass('com.hankcs.hanlp.dictionary.CustomDictionary')
Segmentor = SafeJClass('com.hankcs.hanlp.HanLP')
Segmentor.Config = JClass('com.hankcs.hanlp.HanLP$Config')
PerceptronLexicalAnalyzer = SafeJClass('com.hankcs.hanlp.model.perceptron.PerceptronLexicalAnalyzer')
DoubleArrayTrieSegment = SafeJClass('com.hankcs.hanlp.seg.Other.DoubleArrayTrieSegment')
AhoCorasickDoubleArrayTrie = SafeJClass('com.hankcs.hanlp.collection.AhoCorasick.AhoCorasickDoubleArrayTrie')
IOUtil = SafeJClass('com.hankcs.hanlp.corpus.io.IOUtil')
'''
