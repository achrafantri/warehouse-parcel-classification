package com.example.achats.repository;

import com.example.achats.model.Produit;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

/**
 * Repository bech nod5lou lel donnée eli fel prodits
 * <p>
 * sta3malna feha jpa bech CRUD (create read update delete) yet3mal automatique

 * </p>
 */
@Repository
public interface ProduitRepository extends JpaRepository<Produit, Long> {
    
    /**
     hedhi 3malneha bech les produits kol chway fi page bech ma no93douch habtin louuuuta ken jew fi nafs lpage
     */
    Page<Produit> findByCategorieId(Long categorieId, Pageable pageable);
}
