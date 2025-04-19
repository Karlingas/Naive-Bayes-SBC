import random
import numpy as np

class Node:
    def __init__(self):
        # Indica si el nodo es una hoja, o no
        self.is_leaf = False

        # Atributos relacionados con la variable que representa el nodo
        self.is_num = True      # Indica si la variable es numérica (True) o categórica (False)
        self.cat_dict = None    # Diccionario para variables categóricas con formato {valor: indice}. El "indice" es el del hijo al que lleva la variable de "valor"

        # Atributos cuando el objeto es una raíz
        self.var = None         # Nombre de la variable de corte
        self.var_index = -1     # Índice de la variable de corte. Al pasarselo a x tal que x[var_index], devuelve el valor de la variable, sea un Int o un String
        self.cut_value = 0      # Valor de la variable de corte, en caso de ser numérica
        self.children = []      # Lista de hijos
        self.qvalues = [None]*4     # Lista de valores de los cuartiles. Si es None, no ha sido acortada.


        # Atributos cuando el objeto es una hoja
        self.class_value = -1       # Valor de la clase si el nodo es hoja
        self.class_count = (0,0)    # Tupla con el formato (casos con valor class_value, casos totales en la hoja)

        # Profundidad del nodo
        self.depth = -1

    def __str__(self):
        output = ''
        if(self.is_leaf):
            output += 'Class value: ' + str(self.class_value) + '\tCounts: ' + str(self.class_count)
        else:
            output += 'Feature '+ str(self.var)
            for i in range(len(self.children)):
                output += '\n'+'\t'*(self.depth+1)+str(self.cut_value)+': '+str(self.children[i]) 
            
        return output
    
    # Esta función nos servirá para hacer predicciones recursivamente hasta llegar a un nodo hoja. Debe ser completada
    def predict(self, x):
        if self.is_leaf:
            return self.class_value
        else:
            if self.is_num:
                # Comparar con `cut_value` en nodos numéricos
                if x[self.var_index] <= self.cut_value:
                    return self.children[0].predict(x)
                else:
                    return self.children[1].predict(x)
            # Falta añadir opcion cuando es cut, es decir numerica acortada por cuartiles.
            elif self.qvalues[0] != None:
                if x[self.var_index] <= self.qvalues[0]:
                    return self.children[0].predict(x)
                elif (x[self.var_index] > self.qvalues[0]) and (x[self.var_index] <= self.qvalues[1]):
                    return self.children[1].predict(x)
                elif (x[self.var_index] > self.qvalues[1]) and (x[self.var_index] <= self.qvalues[2]):
                    return self.children[2].predict(x)
                else:
                    return self.children[3].predict(x)
            else:
                # Al pasarselo a x tal que x[var_index], devuelve el valor de la variable
                if x[self.var_index] in self.cat_dict:
                    return self.children[self.cat_dict[x[self.var_index]]].predict(x)
                else:
                    return self.class_value  # Fallback: devolver la clase mayoritaria
                


import time
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from collections import Counter
from sklearn import tree

class C45Classifier(BaseEstimator, ClassifierMixin):

    # Constructor de la clase, aquí se definen e inicializan las variables de la clase.
    def __init__(self, vars, disc, cont, max_depth=2, criterion='classification_error', prune=False):
        self.max_depth = max_depth
        self.criterion = criterion
        self.prune = prune

        self.vars = vars
        self.disc = disc
        self.cont = cont
        self.quartile = set()
        self.qvalues = [None]*len(self.vars)

        # Diccionario que nos permitirá convertir el nombre de la variable en su índice.
        self.features_dict = {feat: i for i, feat in enumerate(self.vars)}

        # Raíz del árbol
        self.tree = Node()  

        self.X_vars = [[]]*len(vars)
        self.y_vars = None
        self.LIMIT = 1000
        
        # Tiempos para estadísticas
        self.t_exec = None
        self.t_exec_float = 0
        self.t_prune = None
        self.t_prune_float = 0
        self.skipped = 0


    # Función para entrenar el modelo.
    def fit(self, X, y):
        t_start = time.time()
        # Guardamos las variables en caso de que falten en la discreta
        for i in range(len(X[0])):
            self.X_vars[i] = np.unique(X[:,i])
        self.y_vars = np.unique(y)
        # Llamada a la función recursiva que aprende el árbol.
        self._partial_fit(X, y, self.tree, 0, set([]))
        self.t_exec_float = time.time() - t_start
        self.t_exec = self.showTime(self.t_exec_float)

        if self.prune:
            t_prune_started = time.time()
            self._prune_tree(X, y, self.tree)
            self.t_prune_float = time.time() - t_prune_started
            self.t_prune = self.showTime(self.t_prune_float)
        
        return self
    
    def showTime(self, time):  # Para imprimir los tiempos con un formato bonito
        hours = int(time // 3600)
        minutes = int((time % 3600) // 60)
        seconds = int(time % 60)
        ms = int((time - int(time)) * 1000000)
        return f"{hours:01d}:{minutes:02d}:{seconds:02d}.{ms:06d}"
    
    # Función para hacer predicciones.
    def predict(self, X):
        return np.array([self.tree.predict(x) for x in X])
    

    # Función recursiva que busca la variable y corte que maximiza la ganancia de información.
    # - Las variables continuas se tratan con un corte binario, lo que quiere decir que pueden ser usadas multiples veces. 
    # - Las variables discretas ramifican tantas veces como valores tengan, asi que solo pueden ser usadas una vez por camino, 
    #   debiendo almacenarlas en el conjunto `borradas`. 
    def _partial_fit(self, X, y, current_tree, current_depth, borradas):
        def _make_leaf():
            current_tree.is_leaf = True
            counts = Counter(y)
            max_value = counts.most_common(1) # most_common(1) devuelve una lista con el elemento más común y su frecuencia.
            if len(max_value):
                current_tree.class_value = max_value[0][0]
                current_tree.class_count = (max_value[0][1], len(y))
            else:
                current_tree.class_value = self.y_vars[0]
                current_tree.class_count = (0, len(y))
            return
            
        # RECORDATORIO: X -> Matriz que contiene en cada fila atributos relacionados con una variable (cada fila representa una variable)
        # y -> vector que contiene la etiqueta para la variable, el nombre o string.
        # Es decir, la fila 0 de X tiene los atributos de la variable con nombre almacenado en y[0].

        # Antes de nada, si hemos alcanzado la profundidad máxima, el nodo se convierte en hoja.
        if current_depth >= self.max_depth:
            _make_leaf()
            return current_tree

        # Primero obtenemos el mejor punto de corte para el nodo actual dependiendo del criterio.
        best_var, cut_value, is_num, is_cut = self._split(X, y, borradas, self.criterion)

        # Si no hay ninguna partición que mejore la actual, el nodo se convierte en hoja.
        if best_var is None:
            _make_leaf()
            return current_tree
    
        # Antes de llamar a la función recursiva, hay que actualizar los valores del árbol (los valores de Node()).
        borradas_copy = borradas.copy()
        if not is_num:    # Solo borramos las variables categóricas ya que estarán totalmente particionadas.
            if is_cut:
                current_tree.qvalues = self.qvalues[self.features_dict[best_var]]
            else:
                borradas_copy.add(best_var)
            current_tree.is_num = False
        else:
            current_tree.cut_value = cut_value # Al ponerlo hace que el error sea 1.0.
               
        current_tree.is_leaf = False
        current_tree.depth = current_depth
        current_tree.var = best_var
        current_tree.var_index = self.features_dict[best_var]
        
        counts = Counter(y)
        max_value = counts.most_common(1) # most_common(1) devuelve una lista con el elemento más común y su frecuencia.
        if len(max_value):
            current_tree.class_count = (max_value[0][1], len(y))
        else:
            current_tree.class_count = (0, len(y))

        # Finalmente, se hace la llamada recursiva en función de si es numérica o categórica.
        if is_num:
            # Dividimos en izquierda y derecha los ÍNDICES desde el valor obtenido de split.
            left_side = X[:, current_tree.var_index] <= cut_value # Máscara de bools que cumplen la condición en la columna seleccionada.
            right_side = X[:, current_tree.var_index] > cut_value

            # Árbol izquierdo
            child1 = self._partial_fit(X[left_side], y[left_side], 
                     Node(), current_depth + 1, 
                     borradas_copy)
            
            # Árbol derecho
            child2 = self._partial_fit(X[right_side], y[right_side],
                     Node(), current_depth + 1,
                     borradas_copy)
            current_tree.children = [child1, child2]
            
        elif is_cut:
            # Si ha sido dividida por cuartiles creamos 4 hijos para acomodar los 4 intervalos
            are_the_variables = X[:, current_tree.var_index] <= self.qvalues[current_tree.var_index][0]
            child = self._partial_fit(X[are_the_variables], y[are_the_variables], Node(), current_depth + 1, borradas_copy)
            current_tree.children.append(child)

            are_the_variables = (X[:, current_tree.var_index] > self.qvalues[current_tree.var_index][0]) & (X[:, current_tree.var_index] <= self.qvalues[current_tree.var_index][1])
            child = self._partial_fit(X[are_the_variables], y[are_the_variables], Node(), current_depth + 1, borradas_copy)
            current_tree.children.append(child)

            are_the_variables = (X[:, current_tree.var_index] > self.qvalues[current_tree.var_index][1]) & (X[:, current_tree.var_index] <= self.qvalues[current_tree.var_index][2])
            child = self._partial_fit(X[are_the_variables], y[are_the_variables], Node(), current_depth + 1, borradas_copy)
            current_tree.children.append(child)

            are_the_variables = X[:, current_tree.var_index] > self.qvalues[current_tree.var_index][2]
            child = self._partial_fit(X[are_the_variables], y[are_the_variables], Node(), current_depth + 1, borradas_copy)
            current_tree.children.append(child)

        else:                
            # Creamos un subarbol por cada variable categórica.
            values = self.X_vars[current_tree.var_index] #np.unique(X[:, current_tree.var_index])
            #current_tree.children = [None]*len(self.features_dict)
            for val in values:
                
                are_the_variables = X[:, current_tree.var_index] == val
                child = self._partial_fit(X[are_the_variables], y[are_the_variables], Node(), current_depth + 1, borradas_copy)
                
                # Guardar el hijo en la lista de children
                current_tree.children.append(child)
                
                # Asegurar que el diccionario `cat_dict` existe y asigna correctamente los valores
                if current_tree.cat_dict is None:
                    current_tree.cat_dict = {}
                current_tree.cat_dict[val] = len(current_tree.children) - 1  # Índice del hijo en la lista
      
        return current_tree


    # Cálculo del mejor punto de corte en función de: Error de clasificación.
    def _split(self, X, y, borradas, criterion='classification_error'):
        # Error actual (sin partición)
        error_best = self._compute_split_criterion(y, criterion)

        is_num = False
        best_var = None
        cut_value =  None  # Para variables categóricas no hay valor de corte (devolvemos None).
        is_cut = False # Para variables numéricas acortadas mediante cuartiles
        
        for var in self.vars:
            index = self.features_dict[var] # Índice de la variable en la matriz X.
            values = np.unique(X[:, index]) # Devuelve los valores de la columna seleccionada.
            if var in self.disc:
                # En X tenemos los valores de las variables, es decir en var estaría Tiempo, y en X en su columna estaría Soleado, Nublado, LLuvioso. En Y esta la variable que queremos predir, por ejemplo Dia bueno o malo, en Y estaría bueno o malo. O dinero en el banco en var, y en X 55, 23, 77
                if var not in borradas:
                    #is_num = False
                    error = 0
                    for val in values:
                        are_the_variables = X[:, index] == val # Máscara de bools que cumplen la condición en la columna seleccionada.
                        error = error + (self._compute_split_criterion(y[are_the_variables], criterion) * (len(y[are_the_variables])/len(y)))

                    # Si el error es mejor que el actual, actualizamos.
                    if error < error_best:
                        error_best = error
                        best_var = var    
                        is_num = False  
                        is_cut = False 
                        cut_value =  None

            elif var in self.cont:   
                if len(self.X_vars[index]) > self.LIMIT:
                    # Si las variables continuas son mas del limite las calcularlamos mediante rangos de cuartiles.           

                    self.quartile.add(var)
                    self.qvalues[self.features_dict[self.vars[index]]]=(np.quantile(X[:, index], [0.25,0.5,.75]))

                    error = 0

                    are_the_variables = X[:, index] <= self.qvalues[index][0] # Máscara de bools que cumplen la condición en la columna seleccionada.
                    error = error + (self._compute_split_criterion(y[are_the_variables], criterion) * (len(y[are_the_variables])/len(y)))

                    are_the_variables = (X[:, index] > self.qvalues[index][0]) & (X[:, index] <= self.qvalues[index][1])
                    error = error + (self._compute_split_criterion(y[are_the_variables], criterion) * (len(y[are_the_variables])/len(y)))

                    are_the_variables = (X[:, index] > self.qvalues[index][1]) & (X[:, index] <= self.qvalues[index][2])
                    error = error + (self._compute_split_criterion(y[are_the_variables], criterion) * (len(y[are_the_variables])/len(y)))

                    are_the_variables = (X[:, index] > self.qvalues[index][2])
                    error = error + (self._compute_split_criterion(y[are_the_variables], criterion) * (len(y[are_the_variables])/len(y)))          

                    # Si el error es mejor que el actual, actualizamos.
                    if error < error_best:
                        error_best = error
                        best_var = var
                        cut_value = None
                        is_num = False
                        is_cut = True
                else:
                    id_X = np.argsort(X[:, index]) # devuelve un array con los numeros de orden que le corresponden a cada indice idX[2] = 0, significa que X[2] es el menor
                    # repeated = set() # Para evitar evaluar varios valores
                    for i in range(len(id_X)):
                        # hay que tener en cuenta si todos los datos son iguales y no hay punto de corte. 
                        # Si la clase es la misma no tiene sentido partir ahi, solo cuando la clase cambia
                        # Ordenar la lista de valores y ver si hay alguno igual.
                        current = id_X[i]
                        following = id_X[i+1] if i+1 < len(id_X) else None

                        # Un OR resulta en un árbol menos preciso que con un AND, pero es bastante más rápido y no disminuye el error.
                       #if (following != None) and ((X[current, index] == X[following, index]) and (y[current] == y[following])):
                        if (following != None) and ((X[current, index] == X[following, index]) or (y[current] == y[following])):
                            self.skipped += 1
                            continue

                        left_side = X[:, index] <=  X[current, index]# ((values[i] + values[i+1]) / 2) # Máscara de bools que cumplen la condición en la columna seleccionada.
                        right_side = X[:, index] > X[current, index] # ((values[i] + values[i+1]) / 2)

                        # Error de clasificación
                        error = ((self._compute_split_criterion(y[left_side], criterion) * (len(y[left_side])/len(y)))
                                + self._compute_split_criterion(y[right_side], criterion) * (len(y[right_side])/len(y)))

                        # Si el error es mejor que el actual, actualizamos.
                        if error < error_best:
                            error_best = error
                            best_var = var
                            cut_value = X[current, index]
                            is_num = True
                            is_cut = False

            # Si conseguimos un error de 0 (óptimo), terminamos
            if error_best == 0:
                break
        if best_var in borradas:
            raise Exception("Error: La variable ya ha sido borrada.")
        return best_var, cut_value, is_num, is_cut

    # Cálculo del mejor punto de corte en función de: Error de clasificación; Entropía; Índice Gini.
    def _compute_split_criterion(self, y, criterion='classification_error'):
        # Completar aquí si tenéis código común a los tres criterios.
        if len(y) == 0: #Si no hay datos, devuelvo error?, devuelvo 0?
            return 0

        if criterion == 'classification_error':
            counts = Counter(y) #almacena el numero de clases y las veces que salen
            most_common = counts.most_common(1)[0][1] #Coge la clase mas comun y coge el numero de veces que ocurre
            return 1 - (most_common / len(y)) #al dividirlo por el len sale la probabilidad de eso, por lo que 1-prob de acierto = prob fallo
        
        elif criterion == 'entropy':
            counts = Counter(y) #almacena el numero de clases y las veces que salen
            probabilidades = np.array(list(counts.values())) / len(y) #cada una de ellas se divide por el numero de veces (probabilidad)
            logaritmos = np.log2(probabilidades) #se calcula el logaritmo en base 2 de cada probabilidad
            return -np.sum(probabilidades * logaritmos) #se multiplica cada probabilidad por su logaritmo, se suman todos y - eso es la entropia

        elif criterion == 'gini':
            numeroClases = Counter(y) #almacena el numero de clases y las veces que salen
            probAlCuadrado = (np.array(list(numeroClases.values())) / len(y))**2 #cada una de ellas se divide por el numero de veces (probabilidad), y luego se eleva al cuadrado
            return 1 - np.sum(probAlCuadrado) #1-SUM p^2
        else:
            raise ValueError('Criterio no válido.')

    
    # Completar esta función para realizar la poda del modelo.
    def _prune_tree(self, X, y, current_tree):
        def calculaProb():
            if current_tree.is_num:
                numerador = current_tree.class_count[1] - current_tree.class_count[0] + 1
                denomi = current_tree.class_count[1] + 2
            else : 
                numerador = current_tree.class_count[1] - current_tree.class_count[0] + len(Counter(y)) - 1
                denomi = current_tree.class_count[1] + len(Counter(y))

            return numerador / denomi
        #Lo primero es ir a cada hoja. Luego recalculamos con la formula y nos vamos al padre. Si ocurre, entonces lo sustituimos y como se convierte en hoja pues repetimos
        def _make_leaf():
            current_tree.is_leaf = True
            counts = Counter(y)
            max_value = counts.most_common(1) # most_common(1) devuelve una lista con el elemento más común y su frecuencia.
            current_tree.class_value = max_value[0][0]
            current_tree.class_count = (max_value[0][1], len(y))
            return

        if not current_tree.is_leaf:
            totalCount = 0
            probPoda = 0
            for hijo in current_tree.children:
                hijito = self._prune_tree( X, y, hijo)
                if hijito == None:
                    return None
                
                totalCount += hijo.class_count[1]

                probPoda += hijito * hijo.class_count[1] 
            
            probPoda = probPoda/totalCount
            
            if calculaProb() < probPoda:
                _make_leaf()
                current_tree.children = []
                
                return probPoda
            
            return None
        
        else:
            return calculaProb()

    # Función para imprimir el modelo.
    def __str__(self):
        return str(self.tree)
    


x_plot = []

def cuentaNodos(nodo):
    sum = len(nodo.children)
    if sum > 0:
        for i in nodo.children:
            sum += cuentaNodos(i)
    
    return sum

def statistics(arbol, X = None, y = None, test = True,  X_test = None, y_test = None):
    ret = f"Error en train:  {arbol.score(X,y)}"
    if test:
        ret += f"\nError en test:  {arbol.score(X_test,y_test)}"
    ret += f"\nNúmero de nodos: {cuentaNodos(arbol.tree)+1}"
    ret += f"\nTiempo de ejecución: {arbol.t_exec}"
    if arbol.prune:
        ret += f"\nTiempo de poda: {arbol.t_prune}"
    ret += f"\nValores continuos saltados (OR): {arbol.skipped}"
    if len(arbol.quartile):
        ret += f"\nVariables discretas tratadas como cuartiles: {arbol.quartile}"
    return f"{ret}"